"""
Implementation of the 'Fitting-Based Sliding Window' (FSW) algorithm for PPG signals.

Reference:
    Zhang, G., et al. (2020).
    "A Noninvasive Blood Glucose Monitoring System Based on Smartphone PPG Signal
    Processing and Machine Learning."
    IEEE Transactions on Industrial Informatics.

Methodology implemented (Algorithm 1):
    1. Get main frequency (f)
    2. Set window size = 0.7 * (1/f)
    3. Slide window to find local minimums (valleys)
    4. Fit baseline using cubic spline interpolation
    5. Find regions where baseline overshoots the signal and extract local minimums
    6. Iteratively refine baseline
    7. Remove baseline from signal
    8. Segment the signal using the valley indices
"""

import numpy as np
from scipy.interpolate import CubicSpline


def _fsw_algorithm(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Algorithm 1: Fitting-Based Sliding Window (FSW)
    """
    N = len(signal)
    if N == 0:
        return signal

    # 1: Get main frequency, f
    yf = np.abs(np.fft.fft(signal))
    xf = np.fft.fftfreq(N, 1 / fs)

    # Isolate typical human heart rate frequencies (0.5 Hz - 4 Hz)
    valid_idx = (xf > 0.5) & (xf < 4.0)
    if np.any(valid_idx):
        f = xf[valid_idx][np.argmax(yf[valid_idx])]
    else:
        f = 1.0  # Fallback

    # 2: Set a window size = 0.7 * 1/f
    window_size = int(0.7 * (1.0 / f) * fs)
    if window_size < 1:
        window_size = 1

    # 3: Slide the window to find minimum of each window.
    # The values and indices are recorded as y and x.
    x = []
    y = []
    for i in range(0, N, window_size):
        end = min(i + window_size, N)
        win_signal = signal[i:end]
        min_idx = i + np.argmin(win_signal)
        x.append(min_idx)
        y.append(signal[min_idx])

    x = np.array(x)
    y = np.array(y)

    # Edge handling to prevent CubicSpline extrapolation errors
    if len(x) > 0 and x[0] != 0:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, signal[0])
    if len(x) > 0 and x[-1] != N - 1:
        x = np.append(x, N - 1)
        y = np.append(y, signal[-1])

    # 6: Iterate steps 2-5
    max_iter = 5
    for _ in range(max_iter):
        # Ensure unique x coordinates for Spline
        _, unique_indices = np.unique(x, return_index=True)
        x_u = x[unique_indices]
        y_u = y[unique_indices]

        if len(x_u) < 4:
            # Not enough points for a cubic spline
            baseline = np.full(N, np.mean(signal))
            break

        # 4: Utilize cubic spline function to fit baseline
        cs = CubicSpline(x_u, y_u)
        baseline = cs(np.arange(N))

        # 5: Separate the point in the signal that matches baseline - signal > 0
        diff = baseline - signal
        if not np.any(diff > 0):
            break  # If no points are > 0, we found the true envelope

        updated = False
        for i in range(1, len(x) - 1):
            # Search window around the current minimum
            start = max(0, x[i] - window_size // 2)
            end = min(N, x[i] + window_size // 2)

            # find the smallest amplitude m, replace x and y
            if np.any(baseline[start:end] > signal[start:end]):
                new_min_idx = start + np.argmin(signal[start:end])
                if new_min_idx != x[i]:
                    x[i] = new_min_idx
                    y[i] = signal[new_min_idx]
                    updated = True

        if not updated:
            break

    # 7: signal - baseline;
    detrended_signal = signal - baseline

    # 8 & 9: Segment n into ni according to x; return ni;
    return detrended_signal
