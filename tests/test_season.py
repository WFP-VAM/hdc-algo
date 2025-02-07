from datetime import datetime

import pandas as pd

from hdc.algo.season import Season

# pylint: disable=use-implicit-booleaness-not-comparison


def test_season():
    # Test initialization and basic attributes
    season = Season([(1, 10), (11, 20), (21, 30)])

    # Check season lengths
    assert len(season) == 3
    assert season.season_lengths == [10, 10, 10]

    # Test raw season ranges
    assert season.raw == [(1, 10), (11, 20), (21, 30)]

    # Test the string representation and hashing
    assert (
        repr(season)
        == "Season(season_ranges=[(1, 10), (11, 20), (21, 30)], season_lengths=[10, 10, 10])"
    )
    assert hash(season) == hash(tuple(season.season_ranges))

    # Test equality
    season_2 = Season([(1, 10), (11, 20), (21, 30)])
    assert season == season_2
    assert season != Season([(1, 10), (11, 20)])
    assert season != Season([(1, 5), (6, 10), (11, 15)])

    # Test season index (dekad of the year)
    assert season.season_index(5) == 1
    assert season.season_index(15) == 2
    assert season.season_index(25) == 3
    assert season.season_index(36) is None
    assert season.season_index(37) is None

    # Test season idx method
    assert season.idx(datetime(2022, 3, 10)) == 1
    assert season.idx(datetime(2022, 6, 15)) == 2
    assert season.idx(datetime(2022, 8, 25)) == 3
    assert season.idx(datetime(2022, 12, 3)) is None

    # Test season label
    assert season.season_label(datetime(2022, 3, 10)) == "2022-01"
    assert season.season_label(datetime(2022, 6, 15)) == "2022-02"
    assert season.season_label(datetime(2022, 8, 25)) == "2022-03"
    assert pd.isna(season.season_label(datetime(2022, 12, 3)))

    # Test cross-year season handling
    season_cross_year = Season([(27, 15)])
    assert len(season_cross_year) == 1
    assert season_cross_year.season_lengths == [36 - 27 + 1 + 15]
    assert season_cross_year != Season([(15, 27)])
    assert season_cross_year.season_index(30) == 1
    assert season_cross_year.season_index(3) == 1
    assert season_cross_year.season_index(22) is None
    assert season_cross_year.idx(datetime(2024, 11, 12)) == 1
    assert season_cross_year.idx(datetime(2025, 3, 28)) == 1
    assert season_cross_year.idx(datetime(2024, 6, 1)) is None
    assert season_cross_year.season_label(datetime(2024, 11, 12)) == "2024-01"
    assert season_cross_year.season_label(datetime(2025, 3, 28)) == "2024-01"
    assert pd.isna(season_cross_year.season_label(datetime(2024, 6, 1)))

    # Test invalid season ranges (should raise ValueError)
    try:
        Season([(1, 10), (5, 15)])  # Overlapping ranges
    except ValueError as e:
        assert str(e) == "Season range (1, 10) overlaps with (5, 15)."

    try:
        Season([(30, 5), (5, 15)])  # Overlapping ranges
    except ValueError as e:
        assert str(e) == "Season range (5, 15) overlaps with (30, 5)."

    try:
        Season([(40, 20)])  # Invalid dekad range
    except ValueError as e:
        assert str(e) == "Invalid season range: (40, 20). Dekads must be in [1, 36]."
