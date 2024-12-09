import datetime as dt

import polars as pl
import pytest

import iup


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


def test_quantile_forecast_validation():
    with pytest.raises(AssertionError, match="quantile"):
        iup.QuantileForecast(
            {"quantile": [-0.1], "date": [dt.date(2020, 1, 1)], "estimate": [0.0]}
        )


def test_sample_forecast_validation():
    iup.SampleForecast(
        pl.DataFrame(
            {"date": [dt.date(2020, 1, 1)], "estimate": [0.0], "sample_id": 0}
        ).with_columns(pl.col("sample_id").cast(pl.Int64))
    )
