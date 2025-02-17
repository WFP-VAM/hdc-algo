"""Season helper class."""

from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Union, overload

import numpy as np

from .dekad import Dekad


class Season:
    """
    Season.

    Represented as a single string. There can be several seasons within a year.
    String representation of a season:
    ``YYYY-season_index``.

    Where ``season_index`` is the index of the season range the specified date falls into.

    If the date doesn't fall into any specified range, that data is referred to as "OoS" for
    "Out of Season".
    """

    __slots__ = ("_seas", "season_range")

    def __init__(
        self,
        date: Union[str, int, datetime, date, "Dekad"],
        season_range: List[Tuple[int, int]],
    ):
        """
        Initialize the Season class with a date and season ranges.

        Args:
            date (Union[str, int, datetime, date, Dekad]): The date representing the season.
            season_range (List[Tuple[int, int]]): List of (start_dekad, end_dekad) tuples defining
                seasons.
        """
        self.season_range = season_range
        self.validate_season_ranges()

        if isinstance(date, Dekad):
            self._seas = self.season_label(date.start_date)
        elif isinstance(date, str):
            dekad_date = Dekad(date)
            self._seas = self.season_label(dekad_date.start_date)
        elif isinstance(date, (datetime, date)):
            self._seas = self.season_label(date)
        else:
            raise ValueError(
                "Invalid date format. Must be a string, datetime, date, or Dekad."
            )

    def __repr__(self):
        return f"Season(season_range={self.season_range}, id={self._seas})"

    def __hash__(self):
        return hash(self._seas)

    def __eq__(self, other):
        if not isinstance(other, Season):
            return False
        return self._seas == other._seas

    def __lt__(self, other):
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas < other._seas

    def __le__(self, other):
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas <= other._seas

    def __gt__(self, other):
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas > other._seas

    def __ge__(self, other):
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas >= other._seas

    @property
    def id(self) -> str:
        """Returns the unique string representation of the season."""
        return self._seas

    def season_index(self, dekad_of_year: int) -> Optional[int]:
        """
        Returns the season index (e.g., 1, 2, etc.) for the given dekad of the year.

        Args:
            dekad_of_year (int): Dekad index (1-36).

        Returns:
            int: Season index or None if no match.
        """
        for i, (start, end) in enumerate(self.season_range):
            if start <= end:  # Normal case
                if start <= dekad_of_year <= end:
                    return i + 1
            else:  # Cross-year case
                if dekad_of_year >= start or dekad_of_year <= end:
                    return i + 1
        return None

    def season_label(self, date: Union[datetime, date]) -> str:
        """
        Returns the season label (e.g., '2021-01', '2021-02') for the provided date.

        For cross-year seasons (e.g., Oct-May), the label uses the starting year.

        Args:
            date (datetime or date): Input date.

        Returns:
            str: Season label (e.g., '2021-01') or an empty string if no match.
        """
        dekad = Dekad(date).yidx
        season_idx = self.season_index(dekad)
        if season_idx is None:
            return "OoS"

        # Determine correct reference year for cross-year seasons
        for start, end in self.season_range:
            if start <= end:  # Normal case
                if start <= dekad <= end:
                    return f"{date.year}-{season_idx:02d}"
            if dekad >= start:  # Cross-year case
                return f"{date.year}-{season_idx:02d}"
            return f"{date.year - 1}-{season_idx:02d}"
        return "OoS"

    def validate_season_ranges(self):
        """
        Ensures that the season ranges are valid (e.g., dekads are within 1-36) and do not overlap.
        """
        for i, (start, end) in enumerate(self.season_range):
            if not (1 <= start <= 36 and 1 <= end <= 36):
                raise ValueError(
                    f"Invalid season range: ({start}, {end}). Dekads must be in [1, 36]."
                )

            # Check for overlaps with previous ranges
            for j, (other_start, other_end) in enumerate(self.season_range):
                if i != j:  # Avoid comparing the season with itself
                    if (start <= other_end and end >= other_start) or (
                        start < other_start and end >= other_end
                    ):
                        raise ValueError(
                            f"Season range ({start}, {end}) overlaps with ({other_start}, {other_end})."
                        )

    @property
    def start_date(self) -> datetime:
        """Start date as python ``datetime``."""
        if self._seas == "OoS":
            return np.nan
        year, season_idx = map(int, self._seas.split("-"))
        start_dekad = self.season_range[season_idx - 1][0]
        return (Dekad(f"{year}01d1") + start_dekad - 1).start_date

    @property
    def end_date(self) -> datetime:
        """End date as python ``datetime``."""
        if self._seas == "OoS":
            return np.nan
        year, season_idx = map(int, self._seas.split("-"))
        start_dekad, end_dekad = self.season_range[season_idx - 1]
        if end_dekad < start_dekad:
            year += 1  # Cross-year case
        return (Dekad(f"{year}01d1") + end_dekad - 1).end_date

    @property
    def date_range(self) -> Tuple[datetime, datetime]:
        """Start and end dates as python ``datetime``."""
        return self.start_date, self.end_date

    @property
    def ndays(self) -> int:
        """Number of days in season."""
        if isinstance(self.start_date, float):
            return -1
        return (self.end_date - self.start_date + timedelta(microseconds=1)).days

    def __radd__(self, n: int) -> "Season":
        """Addition with integer (adds years)."""
        new_dekad = Dekad(self.start_date) + n * 36
        return Season(new_dekad, self.season_range)

    def __add__(self, n: int) -> "Season":
        """Addition with integer (adds years)."""
        new_dekad = Dekad(self.start_date) + n * 36
        return Season(new_dekad, self.season_range)

    @overload
    def __sub__(self, other: int) -> "Season":
        """Subtraction with integer (subtracts years)."""

    @overload
    def __sub__(self, other: "Season") -> int:
        """Subtraction with Season (returns difference in years)."""

    def __sub__(self, other: Union[int, "Season"]) -> Union["Season", int]:
        """Subtraction with integer|Season."""
        if isinstance(other, int):
            new_date = Dekad(self.start_date) - other * 36
            return Season(new_date, self.season_range)
        return self.start_date.year - other.start_date.year
