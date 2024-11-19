import iup
import polars as pl
import pytest
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = (
        pl.DataFrame(
            {
                "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
                "date": [
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
        .with_columns(daily=(pl.col("estimate") / pl.col("interval")).fill_null(0))
        .with_columns(previous=pl.col("daily").shift(1).over("geography"))
        .with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    )

    frame = iup.IncidentUptakeData(frame)

    return frame


def test_extract_starting_conditions(frame):
    """
    Use the last date for each grouping factor
    """
    output = iup.LinearIncidentUptakeModel.extract_starting_conditions(
        frame, ("geography",)
    )

    assert output.sort("geography").equals(
        pl.DataFrame(
            {
                "geography": ["USA", "PA"],
                "last_date": [dt.date(2020, 1, 21), dt.date(2020, 1, 21)],
                "last_daily": [4.0 / 7, 0.4 / 7],
                "last_elapsed": [21, 21],
                "last_cumulative": [8.0, 0.8],
            }
        ).sort("geography")
    )


def test_extract_standards(frame):
    """
    Make a dictionary with as many keys as provided variables,
    and two values per sub-dictionary
    """
    output = iup.LinearIncidentUptakeModel.extract_standards(
        frame, ("estimate", "elapsed")
    )

    correct = {
        "estimate": {"mean": frame["estimate"].mean(), "std": frame["estimate"].std()},
        "elapsed": {"mean": frame["elapsed"].mean(), "std": frame["elapsed"].std()},
    }

    assert output == correct


def test_fit(frame):
    """
    Model should fit a line to two points, giving a 'perfect' fit
    """

    output = iup.LinearIncidentUptakeModel()
    output = output.fit(frame, ("geography",))

    assert output.model.score(output.x, output.y) == 1.0


def test_build_scaffold_handles_no_groups():
    """
    Set up a data frame with 5 rows and 7 columns if no grouping factors are given
    """
    start = pl.DataFrame(
        {
            "last_date": dt.date(2020, 1, 31),
            "last_daily": 0.1,
            "last_elapsed": 31,
            "last_cumulative": 4.0,
            "last_interval": 1,
        }
    )
    start_date = dt.date(2020, 2, 1)
    end_date = dt.date(2020, 2, 29)
    interval = "7d"
    group_cols = None

    output = iup.LinearIncidentUptakeModel.build_scaffold(
        start, start_date, end_date, interval, group_cols
    )

    assert output.shape == (5, 7)


def test_build_scaffold_handles_groups():
    """
    Set up a data frame with 10 rows and 8 columns if a grouping factor with 2 values is given
    """
    start = pl.DataFrame(
        {
            "last_date": dt.date(2020, 1, 31),
            "last_daily": 0.1,
            "last_elapsed": 31,
            "last_cumulative": 4.0,
            "last_interval": 1,
        }
    ).join(pl.DataFrame({"geography": ["USA", "PA"]}), how="cross")
    start_date = dt.date(2020, 2, 1)
    end_date = dt.date(2020, 2, 29)
    interval = "7d"
    group_cols = ("geography",)

    output = iup.LinearIncidentUptakeModel.build_scaffold(
        start, start_date, end_date, interval, group_cols
    )

    assert output.shape == (10, 8)


def test_project_sequentially():
    """
    Model with coef 0 for previous and 1 for elapsed gives elapsed back
    """
    elapsed = tuple([10.0, 17.0])
    start = 3
    standards = {
        "previous": {"mean": 0, "std": 1},
        "elapsed": {"mean": 0, "std": 1},
        "daily": {"mean": 0, "std": 1},
    }
    model = LinearRegression()
    x = np.reshape(np.array([0, 0, 0, 0, 1, 0]), (2, 3))
    y = np.reshape(np.array([0, 1]), (2, 1))
    model.fit(x, y)

    output = iup.LinearIncidentUptakeModel.project_sequentially(
        elapsed, start, standards, model
    )

    assert all(
        np.round(np.delete(output, 0), decimals=10)
        == np.round(np.array(elapsed), decimals=10)
    )
