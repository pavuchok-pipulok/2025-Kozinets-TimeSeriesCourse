import numpy as np

from modules.utils import apply_exclusion_zone

def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    mp = matrix_profile["mp"].copy()
    mpi = matrix_profile["mpi"]

    excl_zone = matrix_profile["excl_zone"]
    if excl_zone is None:
        excl_zone = int(np.ceil(matrix_profile["m"] / 2))

    motifs_idx = []
    motifs_dist = []

    for _ in range(top_k):
        idx = np.nanargmin(mp)
        dist = mp[idx]

        if not np.isfinite(dist):
            break

        nn_idx = int(mpi[idx])

        motifs_idx.append((idx, nn_idx))
        motifs_dist.append(dist)

        # исключаем тривиальные совпадения
        mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, nn_idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
