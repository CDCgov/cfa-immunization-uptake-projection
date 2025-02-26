import datetime as dt

import polars as pl
import pytest

import iup.models


@pytest.fixture
def frame() -> iup.UptakeData:
    """
    Make a mock data frame to uptake data manipulations.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
            "time_end": [
                "2019-12-30",
                "2019-12-30",
                "2020-01-07",
                "2020-01-07",
                "2020-01-14",
                "2020-01-14",
                "2020-01-21",
                "2020-01-21",
            ],
            "estimate": [0.0, 0.0, 0.01, 0.001, 0.03, 0.003, 0.04, 0.004],
            "season": ["2019/2020"] * 8,
        }
    )

    frame = frame.with_columns(time_end=pl.col("time_end").str.to_date("%Y-%m-%d"))

    return iup.UptakeData(frame)


def test_extract_group_combos_handles_groups(frame):
    """
    Find all unique geography-season combos
    """
    frame = iup.models.extract_group_combos(frame, ["geography", "season"])

    assert isinstance(frame, pl.DataFrame)
    assert frame.shape[0] == 2


def test_extract_group_combos_handles_no_groups(frame):
    """
    Returns none since no groups are declared.
    """
    frame = iup.models.extract_group_combos(frame, None)

    assert frame is None


def test_build_scaffold_handles_no_test_data():
    """
    Returns none since no groups are declared.
    """
    start_date = dt.datetime(2020, 1, 3)
    end_date = dt.datetime(2020, 1, 20)
    interval = "7d"
    output = iup.models.build_scaffold(start_date, end_date, interval, None, None, 9, 1)

    assert [d.strftime("%Y-%m-%d") for d in output["time_end"].to_list()] == [
        "2020-01-03",
        "2020-01-10",
        "2020-01-17",
    ]


def test_build_scaffold_handles_test_data(frame):
    """
    Returns none since no groups are declared.
    """
    start_date = dt.datetime(2020, 1, 3)
    end_date = dt.datetime(2020, 1, 20)
    interval = "7d"
    frame = frame.filter(pl.col("geography") == "USA").drop("geography")
    output = iup.models.build_scaffold(
        start_date, end_date, interval, frame, None, 9, 1
    )

    assert [d.strftime("%Y-%m-%d") for d in output["time_end"].to_list()] == [
        "2020-01-07",
        "2020-01-14",
    ]


def test_build_scaffold_handles_groups():
    """
    Returns none since no groups are declared.
    """
    start_date = dt.datetime(2020, 1, 3)
    end_date = dt.datetime(2020, 1, 20)
    interval = "7d"
    group_combos = pl.DataFrame(
        {"geography": ["USA", "PA"], "season": ["2019/2020"] * 2}
    )
    output = iup.models.build_scaffold(
        start_date, end_date, interval, None, group_combos, 9, 1
    )

    assert [d.strftime("%Y-%m-%d") for d in output["time_end"].unique().to_list()] == [
        "2020-01-03",
        "2020-01-10",
        "2020-01-17",
    ]

    assert output.shape[0] == 6
