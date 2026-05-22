from src.models.runners.mean_runner import MeanModelRunner
from src.models.runners.pls_runner import PLSModelRunner
from src.models.runners.resnet34_runner import ResNet34ModelRunner
from src.models.runners.tcn_runner import TCNModelRunner
from src.models.runners.tiny_cnn_runner import TinyCNNModelRunner
from src.models.runners.xgb_runner import XGBModelRunner


def get_model(cfg: dict):
    """Resolve model type to the corresponding runner implementation."""
    mtype = cfg["type"]

    if mtype == "resnet34":
        return ResNet34ModelRunner(cfg["config"])

    if mtype == "xgboost":
        return XGBModelRunner(cfg["config"])

    if mtype == "tiny_cnn":
        return TinyCNNModelRunner(cfg["config"])

    if mtype == "tcn":
        return TCNModelRunner(cfg["config"])

    if mtype == "pls":
        return PLSModelRunner(cfg["config"])

    if mtype == "mean":
        return MeanModelRunner(cfg["config"])

    raise ValueError(f"Unknown model type '{mtype}'")
