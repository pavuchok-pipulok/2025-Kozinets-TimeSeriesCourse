import numpy as np
import stumpy

def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile using STUMPY.

    Note:
    -----
    In stumpy, exclusion zone is implicit and equals m//2 for self-join.
    The parameter `exclusion_zone` is kept only for interface compatibility.
    """

    if ts2 is None:
        # Self-join
        mp = stumpy.stump(
            T_A=ts1,
            m=m,
            ignore_trivial=True
        )
    else:
        # AB-join
        mp = stumpy.stump(
            T_A=ts1,
            T_B=ts2,
            m=m,
            ignore_trivial=False
        )

    return {
        'mp': mp[:, 0],        # matrix profile
        'mpi': mp[:, 1].astype(int),  # matrix profile index
        'm': m,
        'excl_zone': m // 2,   # реальная exclusion zone в STUMPY
        'data': {
            'ts1': ts1,
            'ts2': ts2
        }
    }
