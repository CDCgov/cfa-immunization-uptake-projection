import datetime as dt

import polars as pl

import iup.models


def test_extract_group_combos_handles_groups(frame):
    """
    Find all unique grouping factor combos.
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
    Returns a scaffold using the exact start date.
    """
    start_date = dt.datetime(2020, 1, 3)
    end_date = dt.datetime(2020, 1, 20)
    interval = "7d"
    output = iup.models.build_scaffold(start_date, end_date, interval, None, None, 9, 1)

    assert set([d.strftime("%Y-%m-%d") for d in output["time_end"].to_list()]) == set(
        [
            "2020-01-03",
            "2020-01-10",
            "2020-01-17",
        ]
    )


def test_build_scaffold_handles_test_data(frame):
    """
    Returns a scaffold using the closest start date in the test data.
    """
    start_date = dt.datetime(2020, 1, 3)
    end_date = dt.datetime(2020, 1, 20)
    interval = "7d"
    frame = frame.filter(pl.col("geography") == "USA").drop("geography")
    output = iup.models.build_scaffold(
        start_date, end_date, interval, frame, None, 9, 1
    )

    assert set([d.strftime("%Y-%m-%d") for d in output["time_end"].to_list()]) == set(
        [
            "2020-01-07",
            "2020-01-14",
        ]
    )


def test_build_scaffold_handles_groups():
    """
    Returns a scaffold with dates repeated for different grouping factor combos.
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
