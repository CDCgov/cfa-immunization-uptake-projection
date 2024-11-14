import iup
import polars as pl
import pytest


@pytest.fixture
def frame():
    """
    Make a mock data frame to test data cleaning.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
            "date": [
                "2019-12-30",
                "2019-12-30",
                "2020-01-07",
                "2020-01-07",
                "2020-01-14",
                "2020-01-14",
                "2020-01-21",
                "2020-01-21",
            ],
            "estimate": [0.0, 0.0, 0.1, 1.0, 0.3, 3.0, 0.4, 4.0],
        }
    )

    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    return frame


def test_date_to_season(frame):
    """
    Return the overwinter season, for both fall and spring dates
    """
    output = frame.with_columns(date=iup.UptakeData.date_to_season(pl.col("date")))

    assert all(output["date"] == pl.Series(["2019/2020"] * 8))


def test_date_to_interval(frame):
    """
    Return the interval between dates by grouping factor
    """
    output = frame.with_columns(
        interval=iup.UptakeData.date_to_interval(pl.col("date")).over("geography")
    )

    assert all(
        output["interval"][2:8]
        == pl.Series(
            [
                8.0,
                8.0,
                7.0,
                7.0,
                7.0,
                7.0,
            ]
        )
    )
