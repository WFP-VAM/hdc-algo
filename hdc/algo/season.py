"""Season helper class."""

import datetime
from typing import List, Optional, Tuple, Union, overload

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

        Dekads are represented as integers from 1 to 36, where:
            - Dekad 1 = Jan 1–10
            - Dekad 2 = Jan 11–20
            - ...
            - Dekad 36 = Dec 21–31

        Each season is a tuple of (start_dekad, end_dekad), e.g.:
            (29, 6)  # means from dekad 29 (Oct 11–20) to dekad 6 (Feb 21–28)

        Args:
            date (Union[str, int, datetime, date, Dekad]): The date representing the season.
            season_range (List[Tuple[int, int]]): List of (start_dekad, end_dekad) tuples defining
                seasons: [(start1, end1), (start2, end2), ...].
                The first tuple should refer to the main season, and the second (if any)
                to the secondary season. The ranges will be automatically sorted to ensure this,
                but user input should ideally follow this order.

        Note:
            - The dekad integers are 1-indexed and wrap around the calendar year.
            - For seasons that span two years, we use a wrapping representation:
                e.g., (29, 6) is interpreted as
                "start in year N at dekad 29, end in year N+1 at dekad 6".

        Example:
            season_range = [(7, 17), (29, 6)]  # main season: Mar–Jun, second season: Oct–Feb
        """
        season_range_valid = self.validate_season_ranges(season_range)
        self._season_range = season_range_valid

        if isinstance(date, (str, datetime.date, datetime.datetime, Dekad)):
            self._seas = self.season_label(date)
        elif isinstance(date, int):
            self._seas = date
        else:
            raise ValueError(
                "Invalid date format. Must be a string, datetime, date, or Dekad."
            )

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

    def season_index(
        self, date: Union[str, int, datetime.date, datetime.datetime, "Dekad"]
    ) -> int:
        """
        Return the season index (e.g., 1, 2, etc.) for a given date.

        Args:
            date (Union[str, int, datetime, date, Dekad]): Input date.

        Returns:
            int: Season index or None if no match.
        """
        # Convert date to dekad of the year
        if not isinstance(date, Dekad):
            date = Dekad(date)
        dekad_of_year = date.yidx

        for i, (start, end) in enumerate(self._season_range):
            if start <= end:  # Normal case
                if start <= dekad_of_year <= end:
                    return i + 1
            else:  # Cross-year case
                if dekad_of_year >= start or dekad_of_year <= end:
                    return i + 1
        return -1

    def season_label(
        self, date: Union[str, int, datetime.date, datetime.datetime, "Dekad"]
    ) -> int:
        """
        Return the season label (e.g., 202101, 202102) for the provided date.

        For cross-year seasons (e.g., Oct-May), the label uses the starting year.

        Args:
            date (Union[str, int, datetime, date, Dekad]): Input date.

        Returns:
            str: Season label (e.g., 202101) or an empty string if no match.
        """
        if isinstance(date, Dekad):
            dekad = date
        else:
            dekad = Dekad(date)

        dekad_idx = date.yidx
        season_idx = self.season_index(date)

        if season_idx != -1:
            # Determine correct reference year for cross-year seasons
            for start, end in self._season_range:
                if start <= end:  # Normal case
                    if start <= dekad_idx <= end:
                        return int(f"{dekad.year}{season_idx:02d}")
                if dekad_idx >= start:  # Cross-year case
                    return int(f"{dekad.year}{season_idx:02d}")
                return int(f"{dekad.year - 1}{season_idx:02d}")
        return -1

    def validate_season_ranges(
        self, season_range: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Ensure that the season ranges are valid, sorted, and mutually exclusive.

        - Dekads are integers in [1, 36], 1-indexed.
        - Seasons may wrap into the next year, e.g. (29, 6) is valid (Oct–Feb).
        - Ranges are sorted by start dekad (main season should come first).
        - Ranges must not overlap.
        """
        # Check validity of dekads
        for start, end in season_range:
            if not 1 <= start <= 36 or not 1 <= end <= 36:
                raise ValueError(
                    f"Invalid season range: ({start}, {end}). Dekads must be in [1, 36]."
                )

        # Sort season ranges by start dekad
        season_range = sorted(season_range, key=lambda x: x[0])

        # Convert ranges to sets and check for overlaps
        dekad_sets = []
        for start, end in season_range:
            if start <= end:
                dekads = set(range(start, end + 1))
            else:
                dekads = set(range(start, 37)) | set(range(1, end + 1))
            dekad_sets.append(dekads)

        for i, s1 in enumerate(dekad_sets):
            for j in range(i + 1, len(dekad_sets)):
                s2 = dekad_sets[j]
                if s1 & s2:
                    raise ValueError(
                        f"Season range {season_range[i]} overlaps with {season_range[j]}."
                    )
        return season_range

    @property
    def start_date(self) -> Optional[datetime.datetime]:
        """Start date as python ``datetime``."""
        if self._seas == -1:
            return None
        year = self._seas // 100
        season_idx = self._seas % 100
        start_dekad = self._season_range[season_idx - 1][0]
        return Dekad(36 * year + start_dekad - 1).start_date

    @property
    def end_date(self) -> Optional[datetime.datetime]:
        """End date as python ``datetime``."""
        if self._seas == -1:
            return None
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
        return (
            self.end_date - self.start_date + datetime.timedelta(microseconds=1)
        ).days

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
