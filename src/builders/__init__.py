from . import (
    alghlayini_builder,
    islam_builder,
    prabha_builder,
    sengupta_builder,
    zeynali_builder,
)
from .base import BUILDERS


def get_builder(config):
    """Instantiate a registered builder from benchmark configuration."""
    b_type = config.get("type")

    if b_type not in BUILDERS:
        raise ValueError(
            f"Unkown Builder: '{b_type}'. Available: {list(BUILDERS.keys())}"
        )

    builder_cls = BUILDERS[b_type]
    return builder_cls(config["config"])
