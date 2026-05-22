"""
Implementation of the 'Precision Interval Segmentation' algorithm for PPG signals.

Reference:
    Zeynali, M., Alipour, K., Tarvirdizadeh, B., & Ghamari, M. (2025).
    "Non-invasive blood glucose monitoring using PPG signals with various deep learning models
    and implementation using TinyML."
    Scientific Reports / Nature Portfolio.

Methodology implemented:
    1. Detect Peaks (Local Maxima with adaptive thresholds)
    2. Extract Windows around Peaks (Precision Interval)
    3. Validate Windows (must have exactly 1 peak)
    4. Compute Template (Average of valid windows)
    5. Filter by Cosine Similarity to Template (Quality Assurance)
"""

import numpy as np


def count_peaks(window: np.ndarray, threshold: float = 0.0) -> int:
    """Helper function: Counts peaks in a window (Simple Local Maxima)."""
    peaks = 0
    for i in range(1, len(window) - 1):
        if window[i - 1] < window[i] and window[i] > window[i + 1]:
            if window[i] > threshold:
                peaks += 1
    return peaks


def detect_peaks(
    ppg_signal: np.ndarray, height_threshold: float, distance_threshold: int
) -> list:
    """Implements Eq.(2), (3) and (4) from the paper."""
    peaks = []
    last_peak = -distance_threshold

    # Line 10: for each sample t_i
    for i in range(1, len(ppg_signal) - 1):
        # Eq. (2): Local Maxima Condition
        if ppg_signal[i - 1] < ppg_signal[i] and ppg_signal[i] > ppg_signal[i + 1]:

            # Eq. (3) & (4): Height and Distance Thresholds
            if (
                ppg_signal[i] > height_threshold
                and (i - last_peak) > distance_threshold
            ):
                peaks.append(i)
                last_peak = i
    return peaks


def extract_windows(
    ppg_signal: np.ndarray, peaks: list, window_size: int
) -> tuple[np.ndarray, list]:
    """Extracts windows centered around the detected peaks."""
    windows = []
    valid_indices = []
    half_window = window_size // 2

    for peak in peaks:
        window_start = max(0, peak - half_window)
        window_end = min(len(ppg_signal), peak + half_window)
        window = ppg_signal[window_start:window_end]

        # Technical check: Discard boundary regions (incomplete windows)
        if len(window) != window_size:
            continue
        if count_peaks(window) == 1:
            windows.append(window)
            valid_indices.append(peak)

    return np.array(windows), valid_indices


def compute_template(windows: np.ndarray) -> np.ndarray:
    """Computes the average of all valid windows (Template)."""
    if len(windows) == 0:
        return np.array([])
    return np.mean(windows, axis=0)


def cosine_similarity(window: np.ndarray, template: np.ndarray) -> float:
    """Computes cosine similarity between a specific window and the template."""
    dot_product = np.sum(window * template)
    mag_w = np.sqrt(np.sum(window**2))
    mag_t = np.sqrt(np.sum(template**2))

    if mag_w == 0 or mag_t == 0:
        return 0.0
    return dot_product / (mag_w * mag_t)


def filter_windows(
    windows: np.ndarray, indices: list, template: np.ndarray, threshold: float
):
    """Discards windows that do not match the template (Similarity < Threshold)."""
    filtered_windows = []
    filtered_indices = []

    for i, window in enumerate(windows):
        sim = cosine_similarity(window, template)
        if sim >= threshold:
            filtered_windows.append(window)
            filtered_indices.append(indices[i])

    return np.array(filtered_windows), filtered_indices


def run_segmentation(ppg_signal: np.ndarray, fs=100, params=None):
    """
    Executes the complete Zeynali algorithm on a single PPG signal.

    Args:
        ppg_signal (np.ndarray): The cleaned/filtered PPG signal.
        fs (int): Sampling frequency in Hz.
        params (dict): Configuration dictionary (window_duration, thresholds).

    Returns:
        tuple: (final_windows, final_peak_indices)
    """
    if params is None:
        params = {}

    window_duration = params.get("window_duration", 1.0)
    window_size = int(fs * window_duration)

    h_th = float(params.get("height_threshold", 20.0))
    d_th = int(params.get("distance_threshold", 0.8) * fs)
    sim_th = params.get("similarity_threshold", 0.85)

    peaks = detect_peaks(ppg_signal, h_th, d_th)
    if len(peaks) < 2:
        return np.array([]), []

    windows, valid_indices = extract_windows(ppg_signal, peaks, window_size)
    if len(windows) == 0:
        return np.array([]), []

    template = compute_template(windows)
    filtered_windows, final_indices = filter_windows(
        windows, valid_indices, template, sim_th
    )

    return np.array(filtered_windows), final_indices
