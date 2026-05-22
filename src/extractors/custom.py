from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.signal

from .base import (
    ATTR_EXTRACTOR_TYPE,
    ATTR_SCHEMA_VERSION,
    ATTR_TOTAL_RECS,
    ATTR_TOTAL_SUBJS,
    META_AGE,
    META_BMI,
    META_CASEID,
    META_FS_HZ,
    META_GENDER,
    META_GLUCOSE,
    META_HEIGHT,
    META_SUBJECTID,
    META_WEIGHT,
    BaseExtractor,
    register_extractor,
)

EXTRACTOR_NAME_CUSTOM = "custom_csv_extractor"
VERSION = 1

CONF_INPUT_DIR = "input_dir"
CONF_TARGET_FS = "target_fs"
CONF_ORIG_FS = "orig_fs"
CONF_COL_MAP = "column_mapping"
CONF_TRIM_START_SEC = "trim_start_seconds"
CONF_TRIM_END_SEC = "trim_end_seconds"

KEY_MAP_PPG = "ppg_signal"
KEY_MAP_GLUCOSE = "glucose"
KEY_MAP_ID = "id"
KEY_MAP_AGE = "age"
KEY_MAP_HEIGHT = "height"
KEY_MAP_WEIGHT = "weight"
KEY_MAP_GENDER = "gender"
KEY_MAP_FASTED = "fasted"


DEFAULT_PPG_COL = "ppg"
DEFAULT_GLUCOSE_COL = "Glucose"
DEFAULT_ID_COL = "ID"
DEFAULT_AGE_COL = "Age"
DEFAULT_HEIGHT_COL = "Height"
DEFAULT_WEIGHT_COL = "Weight"
DEFAULT_GENDER_COL = "Gender"
DEFAULT_FASTED_COL = "Fasted"


DIR_LABELS = "Labels"
DIR_RAW = "RawData"
STR_LABEL = "label"
STR_SIGNAL = "signal"


@register_extractor(EXTRACTOR_NAME_CUSTOM)
class CustomCSVExtractor(BaseExtractor):
    def _run_impl(self, out_path: Path) -> Path:
        input_dir = Path(self.config[CONF_INPUT_DIR])
        target_fs = int(self.config.get(CONF_TARGET_FS, 100))
        orig_fs = int(self.config.get(CONF_ORIG_FS, 100))
        col_map = self.config.get(CONF_COL_MAP, {})
        trim_start_sec = float(self.config.get(CONF_TRIM_START_SEC, 0))
        trim_end_sec = float(self.config.get(CONF_TRIM_END_SEC, 0))

        labels_dir = input_dir / DIR_LABELS
        raw_dir = input_dir / DIR_RAW
        csv_label_files = sorted(labels_dir.glob("*.csv"))

        if not csv_label_files:
            self.log(f"No CSV files found in {labels_dir}", level="warning")
            return out_path

        with h5py.File(out_path, "w") as f_out:
            f_out.attrs[ATTR_SCHEMA_VERSION] = VERSION
            f_out.attrs[ATTR_EXTRACTOR_TYPE] = EXTRACTOR_NAME_CUSTOM
            f_out.attrs[CONF_TARGET_FS] = target_fs
            f_out.attrs[CONF_ORIG_FS] = orig_fs
            f_out.attrs[CONF_TRIM_START_SEC] = trim_start_sec
            f_out.attrs[CONF_TRIM_END_SEC] = trim_end_sec

            count = 0
            unique_subjects = set()
            pbar = self.pbar(csv_label_files, desc="Extracting Custom Data")

            for csv_file in pbar:
                try:
                    self.log(
                        f"Processing individual file: {csv_file.name}", level="debug"
                    )
                    df_label = pd.read_csv(csv_file)
                    if df_label.empty:
                        continue
                    row = df_label.iloc[0]

                    base_name = csv_file.stem.lower().replace(STR_LABEL, STR_SIGNAL)
                    sig_csv_path = raw_dir / f"{base_name}.csv"
                    if not sig_csv_path.exists():
                        sig_csv_path = raw_dir / f"{csv_file.name}"
                        if not sig_csv_path.exists():
                            self.log(
                                f"Signal file not found: {base_name}.csv", level="debug"
                            )
                            continue

                    df_sig = pd.read_csv(sig_csv_path)
                    ppg_col = col_map.get(KEY_MAP_PPG, DEFAULT_PPG_COL)
                    if ppg_col not in df_sig.columns:
                        continue

                    sig_raw = df_sig[ppg_col].dropna().values.astype(np.float32)

                    if len(sig_raw) == 0:
                        self.handle_exception(
                            ValueError("No valid PPG data after NaN removal"),
                            context=f"File: {csv_file.name}",
                        )
                        continue

                    if target_fs != orig_fs:
                        samples_target = int(len(sig_raw) * target_fs / orig_fs)
                        sig_ppg = scipy.signal.resample(sig_raw, samples_target)
                    else:
                        sig_ppg = sig_raw

                    trim_start_samples = int(trim_start_sec * target_fs)
                    trim_end_samples = int(trim_end_sec * target_fs)

                    if trim_start_samples > 0:
                        sig_ppg = sig_ppg[trim_start_samples:]

                    if trim_end_samples > 0:
                        sig_ppg = sig_ppg[:-trim_end_samples]

                    caseid_val = row.get(col_map.get(KEY_MAP_ID, DEFAULT_ID_COL), count)
                    caseid = int(caseid_val)

                    h_val = float(
                        row.get(col_map.get(KEY_MAP_HEIGHT, DEFAULT_HEIGHT_COL), 0) or 0
                    )
                    w_val = float(
                        row.get(col_map.get(KEY_MAP_WEIGHT, DEFAULT_WEIGHT_COL), 0) or 0
                    )
                    bmi = w_val / ((h_val / 100.0) ** 2) if h_val > 0 else 0.0

                    meta = {
                        META_GLUCOSE: float(
                            row.get(
                                col_map.get(KEY_MAP_GLUCOSE, DEFAULT_GLUCOSE_COL),
                                np.nan,
                            )
                        ),
                        META_CASEID: caseid,
                        META_SUBJECTID: caseid,
                        META_FS_HZ: target_fs,
                        META_AGE: float(
                            row.get(col_map.get(KEY_MAP_AGE, DEFAULT_AGE_COL), np.nan)
                        ),
                        META_GENDER: (
                            1
                            if str(
                                row.get(
                                    col_map.get(KEY_MAP_GENDER, DEFAULT_GENDER_COL), ""
                                )
                            ).lower()
                            == "male"
                            else 0
                        ),
                        META_HEIGHT: h_val,
                        META_WEIGHT: w_val,
                        META_BMI: bmi,
                        "fasted": int(
                            row.get(col_map.get(KEY_MAP_FASTED, DEFAULT_FASTED_COL), 0)
                        ),
                    }

                    if caseid != -1:
                        unique_subjects.add(caseid)

                    fasted_val = int(
                        row.get(col_map.get(KEY_MAP_FASTED, DEFAULT_FASTED_COL), 0)
                    )
                    fasted_suffix = "_fasted" if fasted_val == 1 else "_postprandial"
                    group_name = f"subject{caseid}{fasted_suffix}"

                    self._save_standardized_group(
                        f_out=f_out, group_name=group_name, ppg=sig_ppg, meta=meta
                    )
                    count += 1

                except Exception as e:
                    self.handle_exception(e, context=f"File: {csv_file.name}")

            f_out.attrs[ATTR_TOTAL_RECS] = int(count)
            f_out.attrs[ATTR_TOTAL_SUBJS] = int(len(unique_subjects))

        return out_path
