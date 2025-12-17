import numpy as np

import numpy as np
from typing import Optional


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    if len(ts1) != len(ts2):
        raise ValueError("Time series must have the same length")

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """
    if len(ts1) != len(ts2):
        raise ValueError("Time series must have the same length")

    n = len(ts1)
    mean1 = np.mean(ts1)
    mean2 = np.mean(ts2)
    std1 = np.std(ts1)
    std2 = np.std(ts2)

    # Normalized Euclidean distance formula
    norm_ed_dist = np.sqrt(
        2 * n * (
                1 - (np.sum((ts1 - mean1) * (ts2 - mean2)) / (n * std1 * std2))
        )
    )
    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1.0) -> float:
    """
    Compute DTW distance between ts1 and ts2 with Sakoe–Chiba band.

    Parameters
    ----------
    ts1 : np.ndarray
        First time series
    ts2 : np.ndarray
        Second time series
    r : float
        Relative width of Sakoe-Chiba band (0..1)

    Returns
    -------
    float
        DTW distance
    """
    n, m = len(ts1), len(ts2)
    # Полоса Сако-Чиба в абсолютных индексах
    window = max(1, int(r * max(n, m)))

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # вставка
                dtw_matrix[i, j - 1],  # удаление
                dtw_matrix[i - 1, j - 1]  # совпадение
            )

    return dtw_matrix[n, m]
