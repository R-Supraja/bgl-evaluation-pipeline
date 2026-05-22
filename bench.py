from __future__ import annotations
from constants import PIPELINE
import os
from pprint import pprint
import tensorflow as tf
from src.benchmark import Benchmark


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.get_logger().setLevel("ERROR")

if __name__ == "__main__":
    df = Benchmark.execute(PIPELINE)
    pprint(df)