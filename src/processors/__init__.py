from . import (
    alghlayini_processor,
    islam_processor,
    prabha_processor,
    sengupta_processor,
    zeynali_processor,
    global_mean_processor,

)
from .base import PROCESSORS


def get_processor(config):
    p_type = config.get("type")

    if p_type not in PROCESSORS:
        raise ValueError(
            f"Unkown Processor-Typ: {p_type}. Available: {list(PROCESSORS.keys())}"
        )

    processor_class = PROCESSORS[p_type]
    return processor_class(config["config"])
