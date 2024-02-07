from typing import List, Tuple

import numpy as np


def to_linspace(x) -> Tuple[np.ndarray, List[int]]:
    """Maps input array to linear space.

    Returns array with linear index (0 - n-1) and list of
    original keys matching the indices.
    """
    keys = np.unique(x)
    keys.sort()
    values = np.arange(keys.size)

    idx = np.searchsorted(keys, x.ravel()).reshape(x.shape)
    idx[idx == keys.size] = 0
    mask = keys[idx] == x.data
    new_pix = np.where(mask, values[idx], 0)

    return new_pix, list(keys)
