"""Test utility functions."""

import numpy as np
import pytest

from hdc.algo.utils import to_linspace


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
