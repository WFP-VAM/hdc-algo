"""Dekad helper class."""

from datetime import date, datetime, timedelta
from typing import Tuple, Union, overload


class Dekad:
    """
    Dekad.

    Represented as a single integer. There are 36 (3*12) dekads in a year.
    Integer representation of a dekad is:
    ``36*year + 3*(month-1) + (dekad_of_the_month - 1)``.

    Where ``month in [1, 12]`` and ``dekad_of_the_month`` is one of ``1,2,3``.
    """

    __slots__ = ("_dkd",)

    def __init__(self, dekad: Union[str, int, datetime, date]):
        """Format ``YYYYMMd{1,2,3}``."""
        if isinstance(dekad, str):
            year, month, idx = int(dekad[:4]), int(dekad[4:6]), int(dekad[-1])
            assert 1 <= month <= 12
            assert 1 <= idx <= 3
            self._dkd = 36 * year + 3 * (month - 1) + (idx - 1)
        elif isinstance(dekad, (date, datetime)):
            d = dekad
            self._dkd = 36 * d.year + 3 * (d.month - 1) + min(2, (d.day - 1) // 10)
        else:
            self._dkd = dekad

    @property
    def year(self) -> int:
        """Extract year."""
        return self._dkd // 36

    @property
    def month(self) -> int:
        """Extract month."""
        return 1 + (self._dkd % 36) // 3

    @property
    def day(self) -> int:
        """Extract day."""
        return 1 + (self._dkd % 3) * 10

    @property
    def idx(self) -> int:
        """Dekad index within a month: ``1,2,3``."""
        return 1 + (self._dkd % 3)

    @property
    def yidx(self) -> int:
        """Dekad index within a year: ``1..36``."""
        return 3 * (self.month - 1) + self.idx

    @property
    def raw(self) -> int:
        """Access raw integer representations."""
        return self._dkd

    def __str__(self):
        """Return string representation."""
        return f"{self.year:04d}{self.month:02d}d{self.idx}"

    def __repr__(self):
        """Return string representation."""
        return f'Dekad("{str(self)}")'

    def __hash__(self):
        """Return Hash."""
        return hash(self._dkd)

    def __eq__(self, other: object) -> bool:
        """Check equality with other Dekad, string or int."""
        if isinstance(other, (str, int, datetime, date)):
            other = Dekad(other)
        if not isinstance(other, Dekad):
            return NotImplemented
        return self._dkd == other._dkd

    def __lt__(self, other: Union[str, int, datetime, date, "Dekad"]) -> bool:
        """Check for less than inequality with other Dekad, string or int."""
        if isinstance(other, (str, int, datetime, date)):
            other = Dekad(other)
        if not isinstance(other, Dekad):
            return NotImplemented
        return self._dkd < other._dkd

    def __gt__(self, other: Union[str, int, datetime, date, "Dekad"]) -> bool:
        """Check for greater than inequality with other Dekad, string or int."""
        if isinstance(other, (str, int, datetime, date)):
            other = Dekad(other)
        if not isinstance(other, Dekad):
            return NotImplemented
        return self._dkd > other._dkd

    def __le__(self, other: Union[str, int, datetime, date, "Dekad"]) -> bool:
        """Check for less than or equal inequality with other Dekad, string or int."""
        if isinstance(other, (str, int, datetime, date)):
            other = Dekad(other)
        if not isinstance(other, Dekad):
            return NotImplemented
        return self._dkd <= other._dkd

    def __ge__(self, other: Union[str, int, datetime, date, "Dekad"]) -> bool:
        """Check for greater than or equal inequality with other Dekad, string or int."""
        if isinstance(other, (str, int, datetime, date)):
            other = Dekad(other)
        if not isinstance(other, Dekad):
            return NotImplemented
        return self._dkd >= other._dkd

    @property
    def start_date(self) -> datetime:
        """Start date as python ``datetime``."""
        return datetime(self.year, self.month, self.day)

    @property
    def end_date(self) -> datetime:
        """End date as python ``datetime``."""
        return (self + 1).start_date - timedelta(microseconds=1)

    @property
    def date_range(self) -> Tuple[datetime, datetime]:
        """Start and end dates as python ``datetime``."""
        return self.start_date, self.end_date

    @property
    def ndays(self) -> int:
        """Number of days in dekad."""
        return (self.end_date - self.start_date + timedelta(microseconds=1)).days

    def __radd__(self, n: int) -> "Dekad":
        """Addition with integer."""
        return Dekad(self._dkd + n)

    def __add__(self, n: int) -> "Dekad":
        """Addition with integer."""
        return Dekad(self._dkd + n)

    @overload
    def __sub__(self, other: int) -> "Dekad":
        """Subtraction with integer."""

    @overload
    def __sub__(self, other: "Dekad") -> int:
        """Subtraction with Dekad."""

    def __sub__(self, other: Union[int, "Dekad"]) -> Union["Dekad", int]:
        """Subtraction with integer|Dekad."""
        if isinstance(other, int):
            return Dekad(self._dkd - other)
        return self._dkd - other._dkd
