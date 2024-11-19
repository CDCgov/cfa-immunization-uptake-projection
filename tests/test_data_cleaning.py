import iup
import polars as pl
import pytest
import datetime as dt


@pytest.fixture
def frame():
    """
    Make a mock data frame to test data cleaning.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA"],
            "date": ["2020-01-07", "2020-01-14", "2020-01-21"],
            "estimate": [0.0, 1.0, 2.0],
            "indicator": ["refused", "booster", "booster"],
        }
    )

    return frame


def test_apply_filters_handles_filters(frame):
    """
    If multiple filters are given to apply_filters, all of them should be applied.
    """
    filters = {"geography": "USA", "indicator": "booster"}

    output = iup.apply_filters(frame, filters)

    assert output.shape[0] == 1
    assert output["estimate"][0] == 2.0


def test_apply_filters_handles_no_filters(frame):
    """
    If no filters are given to apply_filters, the whole frame is returned.
    """
    filters = None

    output = iup.apply_filters(frame, filters)

    assert output.equals(frame)


def test_select_columns_handles_groups(frame):
    """
    If grouping columns are given to select_columns, they are included and renamed.
    """
    estimate_col = "estimate"
    date_col = "date"
    group_cols = {"geography": "region"}
    date_format = "%Y-%m-%d"

    output = iup.select_columns(frame, estimate_col, date_col, group_cols, date_format)

    assert output.shape[1] == 3
    assert "region" in output.columns
    assert output["date"].is_sorted()


def test_select_columns_handles_no_groups(frame):
    """
    If no grouping columns are given to select_columns, they are excluded.
    """
    estimate_col = "estimate"
    date_col = "date"
    group_cols = None
    date_format = "%Y-%m-%d"

    output = iup.select_columns(frame, estimate_col, date_col, group_cols, date_format)

    assert output.shape[1] == 2
    assert "region" not in output.columns
    assert output["date"].is_sorted()


def test_insert_rollout_handles_groups(frame):
    """
    If grouping columns are given to insert_rollout, they are included with rollout.
    """
    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    rollout = dt.date(2020, 1, 1)
    group_cols = {"geography": "region"}
    frame = frame.rename(group_cols).drop("indicator")

    output = iup.insert_rollout(frame, rollout, group_cols)

    assert output.shape[0] == 5
    assert (
        output["date"].value_counts().filter(pl.col("date") == rollout)["count"][0] == 2
    )
    assert output["date"].is_sorted()


def test_insert_rollout_handles_no_groups(frame):
    """
    If grouping columns are given to insert_rollout, they are included with rollout.
    """
    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    rollout = dt.date(2020, 1, 1)
    group_cols = None
    frame = frame.drop(["indicator", "geography"])

    output = iup.insert_rollout(frame, rollout, group_cols)

    assert output.shape[0] == 4
    assert (
        output["date"].value_counts().filter(pl.col("date") == rollout)["count"][0] == 1
    )
    assert output["date"].is_sorted()


def test_extract_group_names_handles_matching_groups():
    """
    If a list of matching group name dictionaries is given, the matching values are returned.
    """
    group_cols = [
        {"geography": "region", "indicator": "outcome"},
        {"geography": "region", "indicator": "outcome"},
    ]

    output = iup.extract_group_names(group_cols)

    assert output == ("region", "outcome")


def test_extract_group_names_handles_unmatched_keys():
    """
    If a list of group name dictionaries with mismatched keys is given, the matching values are returned.
    """
    group_cols = [
        {"geography": "region", "indicator": "outcome"},
        {"area": "region", "indicator": "outcome"},
    ]

    output = iup.extract_group_names(group_cols)

    assert output == ("region", "outcome")


def test_extract_group_names_handles_unmatched_values():
    """
    If a list of group name dictionaries with mismatched values is given, an error is raised.
    """
    group_cols = [
        {"geography": "region", "indicator": "outcome"},
        {"geography": "area", "indicator": "outcome"},
    ]

    with pytest.raises(AssertionError):
        iup.extract_group_names(group_cols)


def test_extract_group_names_handles_no_groups():
    """
    If a group name dictionary is missing, None is returned.
    """
    group_cols = [
        {"geography": "region", "indicator": "outcome"},
        None,
    ]

    output = iup.extract_group_names(group_cols)

    assert output is None
