import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm
    """

    n = len(ts)
    m = len(query)
    N = n - m + 1

    dist_profile = np.zeros(shape=(N,))

    # Z-нормализация запроса (один раз)
    if is_normalize:
        query_norm = z_normalize(query)

    for i in range(N):
        subseq = ts[i:i + m]

        if is_normalize:
            subseq_norm = z_normalize(subseq)
            dist_profile[i] = norm_ED_distance(subseq_norm, query_norm)
        else:
            dist_profile[i] = ED_distance(subseq, query)

    return dist_profile

