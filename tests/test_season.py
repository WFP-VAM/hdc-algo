from datetime import date, datetime, timedelta

from hdc.algo.season import Season

# pylint: disable=use-implicit-booleaness-not-comparison


def test_season():
    # Define season ranges and corresponding dates
    season_ranges = [(1, 10), (11, 20), (21, 30)]
    season_dates = [
        datetime(2022, 1, 10),  # Season 1
        datetime(2022, 4, 15),  # Season 2
        datetime(2022, 7, 25),  # Season 3
    ]

    # Initialize Season instances with corresponding dates
    seasons = [Season(date, season_ranges) for date in season_dates]

    # Test the string representation and hashing
    assert (
        repr(seasons[0])
        == "Season(season_range=[(1, 10), (11, 20), (21, 30)], id=202201)"
    )
    assert hash(seasons[0]) == hash(seasons[0].id)

    # Test equality
    assert seasons[0] == Season(date(2022, 1, 10), season_ranges)
    assert seasons[0] != Season(date(2022, 5, 10), season_ranges)
    assert seasons[0] != Season(date(2022, 12, 10), season_ranges)

    # Test id
    assert seasons[0].id == 202201
    assert seasons[1].id == 202202
    assert seasons[2].id == 202203
    assert Season(datetime(2022, 12, 10), season_ranges).id == -1

    # Test cross-year season handling
    season_cross_year1 = Season(date(2024, 11, 12), [(28, 15)])
    season_cross_year2 = Season(date(2025, 2, 9), [(27, 15)])
    assert season_cross_year1.id == 202401
    assert season_cross_year2.id == 202401
    assert season_cross_year1.season_index(30) == 1
    assert season_cross_year2.season_index(30) == 1
    assert season_cross_year1.season_index(3) == 1
    assert season_cross_year2.season_index(3) == 1
    assert season_cross_year1.season_index(22) is None
    assert season_cross_year2.season_index(22) is None
    assert (season_cross_year1 + 1).id == 202501
    assert (season_cross_year2 + 1).id == 202501
    assert (season_cross_year1 - 1).id == 202301
    assert (season_cross_year2 - 1).id == 202301
    assert season_cross_year1 - season_cross_year2 == 0

    # Test invalid season ranges (should raise ValueError)
    try:
        Season(datetime(2022, 1, 10), [(1, 10), (5, 15)])  # Overlapping ranges
    except ValueError as e:
        assert str(e) == "Season range (1, 10) overlaps with (5, 15)."

    try:
        Season(datetime(2022, 1, 10), [(30, 5), (5, 15)])  # Overlapping ranges
    except ValueError as e:
        assert str(e) == "Season range (5, 15) overlaps with (30, 5)."

    try:
        Season(datetime(2022, 1, 10), [(40, 20)])  # Invalid dekad range
    except ValueError as e:
        assert str(e) == "Invalid season range: (40, 20). Dekads must be in [1, 36]."

    # Test arithmetic operations
    new_season = seasons[0] + 1
    assert new_season.id == 202301

    new_season = seasons[0] - 1
    assert new_season.id == 202101

    year_difference = seasons[1] - seasons[0]
    assert year_difference == 0

    year_difference = seasons[2] - seasons[0]
    assert year_difference == 0

    # Test date range and number of days
    assert seasons[0].date_range == (
        datetime(2022, 1, 1),
        datetime(2022, 4, 11) - timedelta(microseconds=1),
    )
    assert seasons[0].ndays == 100

    # Test cross-year season date range
    assert season_cross_year1.date_range == (
        datetime(2024, 10, 1),
        datetime(2025, 6, 1) - timedelta(microseconds=1),
    )
    assert season_cross_year1.ndays == 243
