import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    """
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start: zone_stop + 1] = np.inf
    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    """
    topK_match_results = {'indices': [], 'distances': []}
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)
        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    """
    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r

    def _calculate_excl_zone(self, m: int) -> int:
        excl_zone = math.ceil(m * self.excl_zone_frac)
        return excl_zone

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        To be implemented in subclasses
        """
        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """
    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        query = copy.deepcopy(query)
        if len(ts_data.shape) != 2:
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        # Вычисляем профиль расстояний
        dist_profile = np.zeros(N, dtype=np.float64)
        query_norm = z_normalize(query) if self.is_normalize else query

        for i in range(N):
            subseq = ts_data[i]
            if self.is_normalize:
                subseq = z_normalize(subseq)
            dist_profile[i] = DTW_distance(subseq, query_norm, r=self.r)

        # Находим top-K совпадений с исключением пересечений
        topk_results = topK_match(dist_profile, excl_zone=excl_zone, topK=self.topK)

        bestmatch = {
            'indices': topk_results['indices'],
            'distances': topk_results['distances']
        }

        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    """
    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        lb_Kim = 0
        # INSERT LB_Kim implementation if needed
        return lb_Kim

    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        lb_Keogh = 0
        # INSERT LB_Keogh implementation if needed
        return lb_Keogh

    def get_statistics(self) -> dict:
        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }
        return statistics

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        query = copy.deepcopy(query)
        if len(ts_data.shape) != 2:
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones(N) * np.inf
        bestmatch = {'index': [], 'distance': []}

        # INSERT UCR-DTW logic if needed

        return bestmatch
