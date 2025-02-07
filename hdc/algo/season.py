"""Season helper class."""

from datetime import date, datetime
from typing import List, Optional, Tuple, Union

import pandas as pd

from .dekad import Dekad


class Season:
    """
    Handles season-based indexing and labeling for time series data using dekads.
    """

    def __init__(self, season_ranges: List[Tuple[int, int]]):
        """
        Initialize the Season class with user-defined season ranges.

        Args:
            season_ranges (List[Tuple[int, int]]): List of (start_dekad, end_dekad) tuples defining seasons.
        """
        self.season_ranges = season_ranges
        self.validate_season_ranges()

    def __repr__(self):
        return f"Season(season_ranges={self.season_ranges}, season_lengths={self.season_lengths})"

    def __hash__(self):
        return hash(tuple(self.season_ranges))

    def __eq__(self, other):
        if not isinstance(other, Season):
            return False
        return self.season_ranges == other.season_ranges

    def __len__(self) -> int:
        return len(self.season_ranges)

    @property
    def season_lengths(self) -> List[int]:
        """
        Returns the length (in dekads) of each season range.

        Returns:
            List[int]: Lengths of each season range (number of dekads in each season).
        """
        lengths = []
        for start, end in self.season_ranges:
            if start <= end:  # Normal season range
                lengths.append(end - start + 1)
            else:  # Cross-year season range
                lengths.append((36 - start + 1) + end)
        return lengths

    @property
    def raw(self) -> List[Tuple[int, int]]:
        """Returns the raw season representation."""
        return self.season_ranges

    def season_index(self, dekad_of_year: int) -> Optional[int]:
        """
        Returns the season index (e.g., 1, 2, etc.) for the given dekad of the year.

        Args:
            dekad_of_year (int): Dekad index (1-36).

        Returns:
            int: Season index or None if no match.
        """
        for i, (start, end) in enumerate(self.season_ranges):
            if start <= end:  # Normal case
                if start <= dekad_of_year <= end:
                    return i + 1
            else:  # Cross-year case
                if dekad_of_year >= start or dekad_of_year <= end:
                    return i + 1
        return None

    def season_label(self, date: Union[datetime, date]) -> Optional[str]:
        """
        Returns the season label (e.g., '2021-01', '2021-02') for the provided date.

        For cross-year seasons (e.g., Oct-May), the label uses the **starting year**.

        Args:
            date (datetime or date): Input date.

        Returns:
            str: Season label (e.g., '2021-01') or NaT if no match.
        """
        dekad = Dekad(date).yidx
        season_idx = self.idx(date)
        if not season_idx:
            return pd.NaT

        # Determine correct reference year for cross-year seasons
        for start, end in self.season_ranges:
            if start <= end:  # Normal case
                if start <= dekad <= end:
                    return f"{date.year}-{season_idx:02d}"
            else:  # Cross-year case
                if dekad >= start or dekad <= end:
                    ref_year = date.year if dekad >= start else date.year - 1
                    return f"{ref_year}-{season_idx:02d}"
        return pd.NaT

    def idx(self, date: Union[datetime, date]) -> Optional[int]:
        """
        Apply season indexing for the provided date and return the season index.

        Args:
            date (datetime or date): Input date.

        Returns:
           int: Season index or None if no match.
        """
        return self.season_index(Dekad(date).yidx)

    def validate_season_ranges(self):
        """
        Ensures that the season ranges are valid (e.g., dekads are within 1-36) and do not overlap.
        """
        for i, (start, end) in enumerate(self.season_ranges):
            if not (1 <= start <= 36 and 1 <= end <= 36):
                raise ValueError(
                    f"Invalid season range: ({start}, {end}). Dekads must be in [1, 36]."
                )

            # Check for overlaps with previous ranges
            for j, (other_start, other_end) in enumerate(self.season_ranges):
                if i != j:  # Avoid comparing the season with itself
                    if (start <= other_end and end >= other_start) or (
                        start < other_start and end >= other_end
                    ):
                        raise ValueError(
                            f"Season range ({start}, {end}) overlaps with ({other_start}, {other_end})."
                        )
