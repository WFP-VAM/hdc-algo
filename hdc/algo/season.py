"""Season helper class."""

import datetime
from typing import List, Tuple, Union, overload

import numpy as np

from .dekad import Dekad


class Season:
    """
    Season.

    Represented as a single integer. There can be several seasons within a year.
    Integer representation of a season:
    ``YYYYseason_index``.

    Where ``season_index`` is the index of the season range the specified date falls into.
    If the date doesn't fall into any specified range, that data is referred to as -1.
    """

    __slots__ = ("_seas", "_season_range")

    def __init__(
        self,
        date: Union[str, int, datetime.date, datetime.datetime, "Dekad"],
        season_range: List[Tuple[int, int]],
    ):
        """
        Initialize the Season class with a date and season ranges.

        Args:
            date (Union[str, int, datetime, date, Dekad]): The date representing the season.
            season_range (List[Tuple[int, int]]): List of (start_dekad, end_dekad) tuples defining
                seasons.
        """
        self._season_range = season_range
        self.validate_season_ranges()

        if isinstance(date, (str, datetime.date, datetime.datetime, Dekad)):
            self._seas = self.season_label(date)
        elif isinstance(date, int):
            self._seas = date
        else:
            raise ValueError("Invalid date format. Must be a string, datetime, date, or Dekad.")

    def __repr__(self):
        """Return string representation."""
        return f"Season(season_range={self.season_range}, id={self._seas})"

    def __hash__(self):
        """Return Hash."""
        return hash(self._seas)

    def __eq__(self, other):
        """Check equality with other Season or int."""
        if not isinstance(other, Season):
            return False
        return self._seas == other._seas

    def __lt__(self, other):
        """Check for less than inequality with other Season or int."""
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas < other._seas

    def __le__(self, other):
        """Check for less than or equal inequality with other Season or int."""
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas <= other._seas

    def __gt__(self, other):
        """Check for greater than inequality with other Season or int."""
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas > other._seas

    def __ge__(self, other):
        """Check for greater than or equal inequality with other Season or int."""
        if not isinstance(other, Season):
            return NotImplemented
        return self._seas >= other._seas

    @property
    def id(self) -> int:
        """Returns the unique integer representation of the season."""
        return self._seas
    
    @property
    def season_range(self) -> List[Tuple[int, int]]:
        """Expose the season range as a read-only property."""
        return self._season_range

    def season_index(self, date: Union[str, int, datetime.date, datetime.datetime, "Dekad"]) -> int:
        """
        Return the season index (e.g., 1, 2, etc.) for a given date.

        Args:
            date (Union[str, int, datetime, date, Dekad]): Input date.

        Returns:
            int: Season index or None if no match.
        """
        # Convert date to dekad of the year
        dekad_of_year = Dekad(date).yidx

        for i, (start, end) in enumerate(self._season_range):
            if start <= end:  # Normal case
                if start <= dekad_of_year <= end:
                    return i + 1
            else:  # Cross-year case
                if dekad_of_year >= start or dekad_of_year <= end:
                    return i + 1
        return -1

    def season_label(self, date: Union[str, int, datetime.date, datetime.datetime, "Dekad"]) -> int:
        """
        Return the season label (e.g., 202101, 202102) for the provided date.

        For cross-year seasons (e.g., Oct-May), the label uses the starting year.

        Args:
            date (Union[str, int, datetime, date, Dekad]): Input date.

        Returns:
            str: Season label (e.g., 202101) or an empty string if no match.
        """
        dekad = Dekad(date).yidx
        season_idx = self.season_index(date)

        if season_idx != -1:
            # Determine correct reference year for cross-year seasons
            for start, end in self._season_range:
                if start <= end:  # Normal case
                    if start <= dekad <= end:
                        return int(f"{date.year}{season_idx:02d}")
                if dekad >= start:  # Cross-year case
                    return int(f"{date.year}{season_idx:02d}")
                return int(f"{date.year - 1}{season_idx:02d}")
        return -1

    def validate_season_ranges(self):
        """Ensure that the season ranges are valid and mutually exclusive."""
        for i, (start, end) in enumerate(self._season_range):
            if not (1 <= start <= 36 and 1 <= end <= 36):
                raise ValueError(
                    f"Invalid season range: ({start}, {end}). Dekads must be in [1, 36]."
                )

            # Check for overlaps with previous ranges
            for j, (start2, other2) in enumerate(self._season_range):
                if i != j:  # Avoid comparing the season with itself
                    if (start <= other2 and end >= start2) or (
                        start < start2 and end >= other2
                    ):
                        raise ValueError(
                            f"Season range ({start}, {end}) overlaps with ({start2}, {other2})."
                        )

    @property
    def start_date(self) -> datetime.datetime:
        """Start date as python ``datetime``."""
        if self._seas == -1:
            return np.nan
        year = self._seas // 100
        season_idx = self._seas % 100
        start_dekad = self._season_range[season_idx - 1][0]
        return Dekad(36 * year + start_dekad - 1).start_date

    @property
    def end_date(self) -> datetime.datetime:
        """End date as python ``datetime``."""
        if self._seas == -1:
            return np.nan
        season_idx = self._seas % 100
        start_dekad, end_dekad = self._season_range[season_idx - 1]
        year = self._seas // 100 + int(end_dekad < start_dekad)
        return Dekad(36 * year + end_dekad - 1).end_date

    @property
    def date_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Start and end dates as python ``datetime``."""
        return self.start_date, self.end_date

    @property
    def ndays(self) -> int:
        """Number of days in season."""
        if isinstance(self.start_date, float):
            return -1
        return (self.end_date - self.start_date + datetime.timedelta(microseconds=1)).days

    def __radd__(self, n: int) -> "Season":
        """Addition with integer (adds years)."""
        return Season(self._seas + n * 100, self._season_range)

    def __add__(self, n: int) -> "Season":
        """Addition with integer (adds years)."""
        return Season(self._seas + n * 100, self._season_range)

    @overload
    def __sub__(self, other: int) -> "Season":
        """Subtraction with integer (subtracts years)."""

    @overload
    def __sub__(self, other: "Season") -> int:
        """Subtraction with Season (returns difference in years)."""

    def __sub__(self, other: Union[int, "Season"]) -> Union["Season", int]:
        """Subtraction with integer|Season."""
        if isinstance(other, int):
            return Season(self._seas - other * 100, self._season_range)
        return self.start_date.year - other.start_date.year
