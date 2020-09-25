import numpy as np


def NSS(saliency_map, xs, ys):
    '''https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py'''
    xs = np.asarray(xs, dtype=np.int)
    ys = np.asarray(ys, dtype=np.int)

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[ys, xs].copy()
    value -= mean

    if std:
        value /= std

    return value


def CC(saliency_map_1, saliency_map_2):
    '''https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py'''
    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]
