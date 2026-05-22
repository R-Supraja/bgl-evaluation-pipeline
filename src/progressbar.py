from __future__ import annotations

import datetime
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


@dataclass
class ProgressConfig:
    enabled: bool = True
    mininterval: float = 0.5
    leave: bool = True


class ProgressReporter:
    """Adds shared logging and progress bar helpers to pipeline classes."""

    _global_log_path: Optional[Path] = None

    def __init__(self, *args, **kwargs):
        """Initializes the reporter state and preserves parent initialization."""
        super().__init__(*args, **kwargs)
        self._current_stage: Optional[str] = None

    @classmethod
    def set_logfile(cls, path: Path):
        """Sets the shared log file path and creates missing directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._global_log_path = path

    def _progress_cfg(self) -> ProgressConfig:
        """Builds a normalized progress configuration from self.config."""
        cfg = getattr(self, "config", {}) or {}
        p = cfg.get("progress", True)
        if isinstance(p, dict):
            return ProgressConfig(
                enabled=bool(p.get("enabled", True)),
                mininterval=float(p.get("mininterval", 0.5)),
                leave=bool(p.get("leave", False)),
            )
        return ProgressConfig(enabled=bool(p))

    def _log_level(self) -> int:
        """Converts the configured log level into a numeric verbosity value."""
        cfg = getattr(self, "config", {}) or {}
        lvl = cfg.get("log_level", "info")
        if isinstance(lvl, int):
            return int(lvl)
        lvl_s = str(lvl).lower()
        if lvl_s in ("quiet", "none", "off", "0"):
            return 0
        if lvl_s in ("debug", "2"):
            return 2
        return 1

    def log(self, msg: str, *, level: str = "info") -> None:
        """Logs a message to the file and optionally echoes it to the console."""
        lvl_limit = self._log_level()
        level_up = level.upper()
        want = 2 if level_up == "DEBUG" else 1

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage = f"[{self._current_stage}] " if self._current_stage else ""

        if level_up == "DEBUG":
            display_msg = f"[{ts}]   └─ {msg}"
        elif level_up in ("ERROR", "WARNING", "CRITICAL"):
            display_msg = f"[{ts}] [{level_up}] {msg}"
        else:
            display_msg = f"[{ts}] {stage}{msg}"

        log_path = ProgressReporter._global_log_path
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{display_msg}\n")
                f.flush()

        if lvl_limit >= want:
            pc = self._progress_cfg()
            if pc.enabled:
                tqdm.write(display_msg)
            else:
                print(display_msg)

    def handle_exception(
        self, e: Exception, context: str = "", show_traceback: bool = False
    ):
        """Logs an exception message and optionally prints its traceback."""
        prefix = f"Error in {context}: " if context else "Error: "
        self.log(f"{prefix}{str(e)}", level="error")
        tb_text = traceback.format_exc()

        log_path = ProgressReporter._global_log_path
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(tb_text + "\n")

        if show_traceback:
            pc = self._progress_cfg()
            if pc.enabled:
                tqdm.write(tb_text)
            else:
                print(tb_text)

    def run_start(
        self, *, step: str, name: str, note: Optional[str] = None, **kwargs
    ) -> float:
        """Logs the start of a step and returns its start timestamp."""
        self._current_stage = self._step_tag(step).upper()
        self.log(
            f"●     [{self._current_stage}] {name} ({self.__class__.__name__})",
            level="info",
        )
        if note:
            self.log(note, level="debug")
        return time.time()

    def run_end(
        self, step: str, *, t0: float, note: Optional[str] = None, **kwargs
    ) -> None:
        """Logs the completion of a step and clears the active stage."""
        tag = self._step_tag(step).upper()
        dt = time.time() - t0
        self.log(f"╰─ [{tag}] DONE {dt:.1f}s", level="info")
        if note:
            self.log(note, level="debug")
        self._current_stage = None

    def _step_tag(self, step: str) -> str:
        """Maps a step name to the short tag used in progress output."""
        s = step.strip().lower()
        mapping = {
            "extractor": "EXTRACT",
            "processor": "PROCESS",
            "builder": "BUILD",
            "model run": "MODEL",
        }
        return mapping.get(s, s.upper())

    def pbar(self, it: Iterable[T], **kwargs):
        """Wraps an iterable in a tqdm progress bar using the active config."""
        pc = self._progress_cfg()
        return tqdm(it, disable=not pc.enabled, mininterval=pc.mininterval, **kwargs)
