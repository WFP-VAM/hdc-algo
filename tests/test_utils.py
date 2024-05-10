"""Test utility functions."""

import numpy as np
import pandas as pd
import pytest

from hdc.algo.utils import get_calibration_indices, to_linspace


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # Test case 1: Array with unique values
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            (np.array([[0, 1, 2], [3, 4, 5]]), [1, 2, 3, 4, 5, 6]),
        ),
        # Test case 2: Array with repeated values
        (
            np.array([[1, 2, 2], [3, 4, 4]]),
            (np.array([[0, 1, 1], [2, 3, 3]]), [1, 2, 3, 4]),
        ),
        # Test case 3: Empty array
        (np.array([]), (np.array([]), [])),
    ],
)
def test_to_linspace(input_array, expected_output):
    x, y = to_linspace(input_array)
    np.testing.assert_equal(x, expected_output[0])
    np.testing.assert_equal(y, expected_output[1])


def test_get_calibration_indices():
    tix = pd.date_range("2010-01-01", "2010-12-31")

    assert get_calibration_indices(tix, (tix[0], tix[-1])) == (0, 365)
    assert get_calibration_indices(tix, ("2010-01-01", "2010-12-31")) == (0, 365)
    assert get_calibration_indices(tix, ("2010-01-15", "2010-12-15")) == (14, 349)

    # groups
    res = np.array(
        [
            [14, 31],
            [0, 28],
            [0, 31],
            [0, 30],
            [0, 31],
            [0, 30],
            [0, 31],
            [0, 31],
            [0, 30],
            [0, 31],
            [0, 30],
            [0, 15],
        ],
        dtype="int16",
    )

    np.testing.assert_array_equal(
        get_calibration_indices(
            tix,
            ("2010-01-15", "2010-12-15"),
            groups=tix.month.values - 1,
            num_groups=12,
        ),
        res,
    )

    np.testing.assert_array_equal(
        get_calibration_indices(
            tix, ("2010-01-15", "2010-12-15"), groups=tix.month.values - 1
        ),
        res,
    )
