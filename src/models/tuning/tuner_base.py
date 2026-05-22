from abc import ABC, abstractmethod
from typing import Any, Dict

from src.models.dataset.base import FoldData


class BaseTuner(ABC):
    """Base interface for fold-level hyperparameter optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_trials = config.get("n_trials", 20)
        self.seed = config.get("seed", -1)
        if self.seed < 0:
            self.seed = None

    @abstractmethod
    def tune(self, fold_data: FoldData, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return best hyperparameters for a given fold and base parameter set."""
        pass
