import numpy as np
from modules.utils import apply_exclusion_zone


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
    mp = matrix_profile['mp'].copy()  # матричный профиль
    excl_zone = matrix_profile['excl_zone']

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    for _ in range(top_k):
        idx = np.argmax(mp)
        dist = mp[idx]
        nn_idx = matrix_profile['mpi'][idx]

        discords_idx.append(idx)
        discords_dist.append(dist)
        discords_nn_idx.append(nn_idx)


        mp = apply_exclusion_zone(mp, idx, excl_zone, val=-np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }
