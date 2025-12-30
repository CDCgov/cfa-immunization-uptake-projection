import datetime as dt

import polars as pl
from polars.testing import assert_frame_equal

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


def test_build_scaffold_handles_no_groups():
    """
    Returns a scaffold using the exact start date.
    """
    test_dates = pl.select(
        pl.date_range(dt.datetime(2020, 1, 3), dt.datetime(2020, 1, 20), "7d").alias(
            "time_end"
        )
    )
    output = iup.models.build_scaffold(test_dates, group_combos=None)

    assert set([d.strftime("%Y-%m-%d") for d in output["time_end"].to_list()]) == set(
        [
            "2020-01-03",
            "2020-01-10",
            "2020-01-17",
        ]
    )


def test_build_scaffold_handles_groups(frame):
    """
    Returns a scaffold with dates repeated for different grouping factor combos.
    """

    group_combos = pl.DataFrame(
        {"geography": ["USA", "PA"], "season": ["2019/2020"] * 2}
    )
    output = iup.models.build_scaffold(test_dates=frame, group_combos=group_combos)

    assert_frame_equal(
        output,
        pl.DataFrame(
            {
                "time_end": [dt.date(2019, 12, 31)] * 2
                + [dt.date(2020, 1, 7)] * 2
                + [dt.date(2020, 1, 14)] * 2
                + [dt.date(2020, 1, 21)] * 2,
                "geography": ["USA", "PA"] * 4,
                "estimate": 0.0,
            }
        ),
        check_row_order=False,
        check_column_order=False,
    )
