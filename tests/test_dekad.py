from datetime import date, datetime, timedelta

from hdc.algo.dekad import Dekad

# pylint: disable=use-implicit-booleaness-not-comparison


def test_dekad():
    eps = timedelta(microseconds=1)
    assert Dekad("202202d1").year == 2022
    assert Dekad("202202d1").month == 2
    assert Dekad("202202d1").idx == 1
    assert Dekad("202202d1").yidx == 4
    assert Dekad("202202d1").start_date == datetime(2022, 2, 1)

    assert Dekad("202202d3").end_date == datetime(2022, 3, 1) - eps

    assert Dekad(date(2021, 2, 28)) == "202102d3"
    assert Dekad(datetime(2021, 3, 31)) == "202103d3"
    assert Dekad(datetime(2021, 3, 21)) == "202103d3"
    assert Dekad(datetime(2021, 3, 20)) == "202103d2"
    assert Dekad(datetime(2021, 3, 10)) == "202103d1"
    assert Dekad(datetime(2021, 1, 1)) == "202101d1"
    assert Dekad(datetime(2021, 12, 11)) == "202112d2"

    # Test equality
    assert Dekad("199912d3") == "199912d3"
    assert Dekad("199912d3") == Dekad(2000 * 36 - 1)
    assert Dekad("202110d3") == 2021 * 36 + 3 * (10 - 1) + (3 - 1)
    assert Dekad(Dekad("199912d3")) == Dekad("199912d3")
    assert (Dekad("200012d3") == []) is False

    # Hashing and representation
    assert len(set([Dekad("20211221")] * 10)) == 1
    assert str(Dekad("199912d3")) == "199912d3"
    assert repr(Dekad("199912d3")) == 'Dekad("199912d3")'

    assert Dekad("199912d3") + 1 == "200001d1"
    assert Dekad("199912d3") - 1 == "199912d2"
    assert Dekad("199912d3") - 3 == "199911d3"
    assert Dekad("199912d1") - 1 == "199911d3"
    assert Dekad("200001d1") - 1 == "199912d3"
    assert Dekad("199912d3") + 36 == "200012d3"

    assert Dekad("202202d3").date_range == (
        datetime(2022, 2, 21),
        datetime(2022, 2, 28, 23, 59, 59, 999999),
    )

    dkd = Dekad("198403d1")
    assert (dkd + 1) == (1 + dkd)
    assert (5 + dkd + 10) - 15 == dkd

    assert (dkd + 5) - dkd == 5
    assert (dkd - 5) - dkd == -5
    assert dkd - dkd == 0

    assert Dekad(dkd.start_date) == dkd
    assert Dekad(dkd.start_date.date()) == dkd
    assert Dekad(dkd.end_date) == dkd
    assert Dekad(dkd.start_date + timedelta(days=3)) == dkd

    assert Dekad("202201d3").ndays == 11
    assert Dekad("202202d1").ndays == 10
    assert Dekad("202202d3").ndays == 8
    assert Dekad("202002d3").ndays == 9

    # Test inequalities
    assert Dekad("202202d3") < Dekad(date(2022, 3, 28))
    assert Dekad("202202d3") < date(2022, 3, 28)
    assert Dekad("202110d3") < "202212d3"
    assert Dekad("202110d3") < 2021 * 36 + 3 * (11 - 1) + (3 - 1)

    assert Dekad("202202d3") > Dekad(date(2019, 3, 28))
    assert Dekad("202202d3") > datetime(2019, 3, 28)
    assert Dekad("202110d3") > "201912d3"
    assert Dekad("202110d3") > 2019 * 36 + 3 * (11 - 1) + (3 - 1)

    assert Dekad("202202d3") >= Dekad(date(2022, 2, 28))
    assert Dekad("202202d3") >= date(2022, 2, 28)
    assert Dekad("202110d3") >= "202110d3"
    assert Dekad("202111d3") >= 2021 * 36 + 3 * (10 - 1) + (3 - 1)

    assert Dekad("202202d3") <= Dekad(date(2023, 2, 28))
    assert Dekad("202202d3") <= date(2023, 2, 28)
    assert Dekad("202110d3") <= "202110d3"
    assert Dekad("202110d3") <= 2021 * 36 + 3 * (10 - 1) + (3 - 1)
