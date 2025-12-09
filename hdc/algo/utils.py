"""hcd-algo utility functions."""

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DateType: TypeAlias = str | pd.Timestamp | np.datetime64


def to_linspace(x) -> tuple[NDArray[(np.int16,)], list[int]]:
    """Map input array to linear space.

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


def get_calibration_indices(
    time: pd.DatetimeIndex,
    calibration_range: tuple[DateType, DateType],
    groups: Iterable[int | float | str] | None = None,
    num_groups: int | None = None,
) -> tuple[int, int] | np.ndarray:
    """
    Get the calibration indices for a given time range.

    This function returns indices for a calibration period (e.g. used for SPI)
    given an index of timestamps and a start & stop date.
    If groups are provided, the indices are returned per group, as an
    array of shape (num_groups, 2) where the first column is the start index and
    the second column is the stop index.

    Parameters
    ----------
        time: The time index.
        start: The start time of the calibration range.
        stop: The stop time of the calibration range.
        groups: Optional groups to consider for calibration.
        num_groups: Optional number of groups to consider for calibration.
    """
    begin, end = calibration_range

    def _get_ix(x: NDArray[(np.datetime64,)], v: DateType, side: str):
        return x.searchsorted(np.datetime64(v), side)  # type: ignore

    if groups is not None:
        if num_groups is None:
            num_groups = len(np.unique(np.array(groups)))
        return np.array(
            [
                [
                    _get_ix(time[groups == ix].values, begin, "left"),  # type: ignore[attr-defined]
                    _get_ix(time[groups == ix].values, end, "right"),  # type: ignore[attr-defined]
                ]
                for ix in range(num_groups)
            ],
            dtype="int16",
        )

    return _get_ix(time.values, begin, "left"), _get_ix(time.values, end, "right")
