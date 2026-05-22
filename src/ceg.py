from __future__ import annotations

from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from error_grids import zone_accuracy


def compute_clarke_zone_accuracies(
    pred_arr: np.ndarray,
    ref_arr: np.ndarray,
) -> Dict[str, float]:
    accuracy_values = zone_accuracy(ref_arr, pred_arr, "clarke")
    accuracy_values = np.asarray(accuracy_values, dtype=float) * 100.0

    zones = {
        "Zone A": float(accuracy_values[0]),
        "Zone B": float(accuracy_values[1]),
        "Zone C": float(accuracy_values[2]),
        "Zone D": float(accuracy_values[3]),
        "Zone E": float(accuracy_values[4]),
    }
    return zones


def print_ceg_zones(
    pred_arr: np.ndarray,
    ref_arr: np.ndarray,
) -> pd.DataFrame:
    zones = compute_clarke_zone_accuracies(pred_arr, ref_arr)
    df = pd.DataFrame([zones])
    return df


def plot_ceg(
    pred_arr: np.ndarray,
    ref_arr: np.ndarray,
    title: str = "Clarke Error Grid",
    point_size: int = 3,
    color: Union[
        str,
        Tuple[float, float, float],
        Tuple[float, float, float, float],
    ] = "green",
    plotsize=10,
    ax=None,
    point_alpha=1.0,
    alpha_helpline=1.0,
    bg_max: int = 400,
) -> None:
    pred = np.asarray(pred_arr, dtype=float)
    ref = np.asarray(ref_arr, dtype=float)
    show_plot_at_end = True

    if ax is None:
        fig, ax = plt.subplots(figsize=(plotsize, plotsize))
        show_plot_at_end = True
    else:
        show_plot_at_end = False

    ax.scatter(
        ref,
        pred,
        # marker="o",
        color=color,
        s=point_size,
        alpha=point_alpha,
    )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Reference BG (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Prediction BG (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, bg_max)

    # ax.grid(True, linestyle="--", alpha=0.6)

    lines_style = "--"

    # y = x
    ax.plot([0, 400], [0, 400], lines_style, color="black", alpha=alpha_helpline)

    # Zone A
    ax.plot([0, 70 / 1.2], [70, 70], lines_style, color="black", alpha=alpha_helpline)
    ax.plot(
        [70 / 1.2, 400],
        [70, 400 * 1.2],
        lines_style,
        color="black",
        alpha=alpha_helpline,
    )

    ax.plot([70, 70], [0, 70 * 0.8], lines_style, color="black", alpha=alpha_helpline)
    ax.plot(
        [70, 400],
        [70 * 0.8, 400 * 0.8],
        lines_style,
        color="black",
        alpha=alpha_helpline,
    )

    # left vertical cut for B, C, D, E
    ax.plot([70, 70], [70 * 1.2, 400], lines_style, color="black", alpha=alpha_helpline)

    # Zone E lower right
    ax.plot([180, 400], [70, 70], lines_style, color="black", alpha=alpha_helpline)
    ax.plot([180, 180], [70, 0], lines_style, color="black", alpha=alpha_helpline)

    # Zone C right lower
    ax.plot([180, 130], [70, 0], lines_style, color="black", alpha=alpha_helpline)

    # Zone D lower
    ax.plot([240, 240], [70, 180], lines_style, color="black", alpha=alpha_helpline)
    ax.plot([240, 400], [180, 180], lines_style, color="black", alpha=alpha_helpline)

    # Zone D and E
    ax.plot([0, 70], [180, 180], lines_style, color="black", alpha=alpha_helpline)

    # Zone upper C and B
    ax.plot([70, 290], [180, 400], lines_style, color="black", alpha=alpha_helpline)

    # Zone labels
    font_size = 20
    alpha_v = 0.7
    ax.text(30, 15, "A", fontsize=font_size, alpha=alpha_v)
    ax.text(350, 320, "A", fontsize=font_size, alpha=alpha_v)
    ax.text(290, 320, "A", fontsize=font_size, alpha=alpha_v)
    ax.text(350, 230, "B", fontsize=font_size, alpha=alpha_v)
    ax.text(240, 320, "B", fontsize=font_size, alpha=alpha_v)
    ax.text(160, 320, "C", fontsize=font_size, alpha=alpha_v)
    ax.text(160, 20, "C", fontsize=font_size, alpha=alpha_v)
    ax.text(30, 140, "D", fontsize=font_size, alpha=alpha_v)
    ax.text(350, 120, "D", fontsize=font_size, alpha=alpha_v)
    ax.text(350, 20, "E", fontsize=font_size, alpha=alpha_v)
    ax.text(30, 320, "E", fontsize=font_size, alpha=alpha_v)

    if show_plot_at_end:
        plt.show()
