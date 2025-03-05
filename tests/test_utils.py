import polars as pl
import pytest

import iup
import iup.utils


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
            "time_end": [
                "2019-12-31",
                "2019-12-31",
                "2020-01-07",
                "2020-01-07",
                "2020-01-14",
                "2020-01-14",
                "2020-01-21",
                "2020-01-21",
            ],
            "estimate": [0.0, 0.0, 1.0, 0.1, 3.0, 0.3, 4.0, 0.4],
            "season": "2019/2020",
            "elapsed": [0, 0, 7, 7, 14, 14, 21, 21],
            "interval": [None, None, 7, 7, 7, 7, 7, 7],
        }
    )

    return frame


def test_extract_standards(frame):
    """
    Make a dictionary with as many keys as provided variables,
    and two values per sub-dictionary
    """
    output = iup.utils.extract_standards(frame, ("estimate", "elapsed"))

    correct = {
        "estimate": {"mean": frame["estimate"].mean(), "std": frame["estimate"].std()},
        "elapsed": {"mean": frame["elapsed"].mean(), "std": frame["elapsed"].std()},
    }

    assert output == correct


def test_date_to_season(frame):
    """
    Season is extracted from a column of dates
    """
    frame = frame.with_columns(
        season2=pl.col("time_end").pipe(iup.utils.date_to_season)
    )

    assert all(frame["season"] == frame["season2"])


def test_date_to_elapsed_handles_no_season_start(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.utils.date_to_elapsed(pl.col("time_end")).over("geography")
    )

    expected = pl.Series([0.0, 0.0, 7.0, 7.0, 14.0, 14.0, 21.0, 21.0])

    assert (output["elapsed"] == expected).all()


def test_date_to_elapsed_handles_season_start(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.utils.date_to_elapsed(pl.col("time_end"), 9, 1).over("geography")
    )

    expected = pl.Series(
        [
            121.0,
            121.0,
            128.0,
            128.0,
            135.0,
            135.0,
            142.0,
            142.0,
        ]
    )

    assert (output["elapsed"] == expected).all()


def test_date_to_interval(frame):
    """
    Return the interval between dates by grouping factor.
    """
    output = frame.sort(["geography", "time_end"]).with_columns(
        interval=iup.utils.date_to_interval(pl.col("time_end")).over("geography")
    )

    expected = pl.Series("interval", ([None] + [7.0] * 3) * 2)

    assert (output["interval"] == expected).all()
