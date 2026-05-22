from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from pybaselines import Baseline

from src.extractors.base import (
    ATTR_ADDITIONAL_SIG,
    ATTR_TOTAL_RECS,
    KEY_PPG_RAW,
    KEY_SPO2_RAW,
    META_FS_HZ,
)
from src.utils import get_group, load_dataset_array

from .base import ATTR_PROCESSOR_TYPE, BaseProcessor, register_processor

PROCESSOR_NAME_ISLAM = "islam_processor"

# Default values used when config does not provide explicit parameters.
DEFAULT_GAUSS_C = 0.068
DEFAULT_GAUSS_W = 0.0543
DEFAULT_ALS_LAM = 1e5
DEFAULT_ALS_P = 0.01


@register_processor(PROCESSOR_NAME_ISLAM)
class IslamPPGFilter(BaseProcessor):
    """Apply Islam et al. (2021) specific filtering to PPG recordings."""

    def __init__(self, config):
        super().__init__(config)
        self.als_lam = float(self.config.get("als_lam", DEFAULT_ALS_LAM))
        self.als_p = float(self.config.get("als_p", DEFAULT_ALS_P))
        # Keep Gaussian constants fixed to the paper values for strict replication.
        self.gauss_c = DEFAULT_GAUSS_C
        self.gauss_w = DEFAULT_GAUSS_W

        if self.als_lam <= 0:
            raise ValueError("islam_processor config 'als_lam' must be > 0.")
        if not (0 < self.als_p < 1):
            raise ValueError("islam_processor config 'als_p' must be in (0, 1).")
        if self.gauss_w <= 0:
            raise ValueError("islam_processor config 'gauss_w' must be > 0.")

    def _gaussian_frequency_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a Gaussian bandpass filter in the frequency domain
        Based on Islam et al.: G(x) = exp(-(x-c)^2 / (2*w^2))
        """
        N = len(signal)
        signal_fft = np.fft.fft(signal)

        freqs = np.fft.fftfreq(N)
        pos_freqs = np.abs(freqs)

        G = np.exp(-((pos_freqs - self.gauss_c) ** 2) / (2 * (self.gauss_w**2)))

        filtered_signal = np.real(np.fft.ifft(signal_fft * G))
        return filtered_signal

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:

            f_out.attrs[ATTR_PROCESSOR_TYPE] = PROCESSOR_NAME_ISLAM
            f_out.attrs["config_als_lam"] = self.als_lam
            f_out.attrs["config_als_p"] = self.als_p
            f_out.attrs["config_gauss_c"] = self.gauss_c
            f_out.attrs["config_gauss_w"] = self.gauss_w

            fs = int(f_in.attrs.get(META_FS_HZ, -1))
            if fs <= 0:
                raise ValueError(f"Invalid or missing sampling rate: {fs}")

            f_out.attrs[META_FS_HZ] = fs
            f_out.attrs[ATTR_ADDITIONAL_SIG] = f_in.attrs.get(ATTR_ADDITIONAL_SIG, [])

            keys = list(f_in.keys())
            total_recordings = len(keys)
            pbar = self.pbar(
                keys,
                desc=f"Processing ({PROCESSOR_NAME_ISLAM})",
                total=len(keys),
                unit="grp",
            )

            removed_count = 0
            removed_quality = 0

            def update_postfix():
                pbar.set_postfix(total=total_recordings, removed=removed_count)

            for key in pbar:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    continue

                try:
                    ppg_raw = load_dataset_array(grp_in, KEY_PPG_RAW, verbose=False)
                    spo2_raw = load_dataset_array(grp_in, KEY_SPO2_RAW, verbose=False)

                    if ppg_raw is None or len(ppg_raw) == 0:
                        removed_count += 1
                        update_postfix()
                        continue

                    # 1. Asymmetric Least Squares (ALS) baseline correction
                    baseline_fitter = Baseline()
                    baseline, _ = baseline_fitter.asls(
                        ppg_raw, lam=self.als_lam, p=self.als_p
                    )
                    ppg_als = ppg_raw - baseline

                    # 2. Gaussian Filter in Frequency Domain
                    ppg_gauss = self._gaussian_frequency_filter(ppg_als)

                    # 3. Normalization from 0 to 1
                    ppg_clean = self._normalize_0_1(ppg_gauss)

                    # Dummy quality handling
                    quality_dummy = np.ones_like(ppg_clean, dtype=np.float32)

                    self.save_processed_group(
                        f_out=f_out,
                        key=key,
                        ppg_clean=ppg_clean,
                        spo2_raw=spo2_raw,
                        quality=quality_dummy,
                        meta_attrs=dict(grp_in.attrs),
                        bad_ratio=0.0,
                    )

                    grp_out = f_out[key]
                    self._copy_additional_signals(grp_in, grp_out)

                except Exception as e:
                    self.handle_exception(e, f"{PROCESSOR_NAME_ISLAM} with {key}")
                    removed_count += 1
                    update_postfix()
                    continue

            f_out.attrs[ATTR_TOTAL_RECS] = int(total_recordings - removed_count)
            f_out.attrs["total_removed_recordings"] = int(removed_count)
            f_out.attrs["total_removed_quality"] = int(removed_quality)

            update_postfix()
        return out_path
