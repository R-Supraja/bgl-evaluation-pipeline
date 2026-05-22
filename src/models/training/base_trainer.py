from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.models.core.model import Model
from src.models.dataset.base import FoldData
from src.models.tuning.tuner_base import BaseTuner


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    fit_verbose: int = 0
    seed: Optional[int] = None
    deterministic: bool = False

    init_from_dir: Optional[str] = None


@dataclass
class TrainingResult:
    val_rmse: float
    history: Optional[Dict[str, Any]] = None

    model_size_kb: Optional[float] = None


class Trainer(ABC):
    def __init__(
        self, model: Model, config: TrainingConfig, tuner: Optional[BaseTuner] = None
    ):
        self.model = model
        self.config = config
        self.tuner = tuner

    @abstractmethod
    def train_fold(
        self,
        fold_data: FoldData,
        fold_idx: int,
    ) -> Tuple[Model, TrainingResult]: ...
