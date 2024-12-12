import datetime as dt

import polars as pl
import pytest

import iup
import iup.models


@pytest.fixture
def frame():
    """
    Make a mock data frame to test data cleaning.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA"],
            "date": ["2020-01-07", "2020-01-14", "2020-01-21"],
            "estimate": [0.0, 0.1, 0.2],
            "indicator": ["refused", "booster", "booster"],
        }
    )

    return frame


def test_insert_rollout_handles_groups(frame):
    """
    If grouping columns are given to insert_rollout, a separate rollout is inserted for each group.
    """
    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    rollout = [dt.date(2020, 1, 1), dt.date(2021, 1, 1)]
    group_cols = ("geography",)
    frame = iup.CumulativeUptakeData(frame.drop("indicator"))

    output = frame.insert_rollout(rollout, group_cols)

    assert output.shape[0] == 7
    assert (
        output["date"].value_counts().filter(pl.col("date") == rollout[0])["count"][0]
        == 2
    )
    assert output["date"].is_sorted()


def test_insert_rollout_handles_no_groups(frame):
    """
    If no grouping columns are given to insert_rollout, only one of each rollout is inserted.
    """
    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    rollout = [dt.date(2020, 1, 1), dt.date(2021, 1, 1)]
    group_cols = None
    frame = iup.CumulativeUptakeData(frame.drop(["indicator", "geography"]))

    output = frame.insert_rollout(rollout, group_cols)

    assert output.shape[0] == 5
    assert (
        output["date"].value_counts().filter(pl.col("date") == rollout[0])["count"][0]
        == 1
    )
    assert output["date"].is_sorted()


def test_extract_group_names_handles_matched_groups():
    """
    If a list of identical group name lists is given, just one tuple is returned.
    """
    group_cols = [["geography", "indicator"], ["geography", "indicator"]]

    output = iup.extract_group_names(group_cols)

    assert output == ("geography", "indicator")


def test_extract_group_names_handles_unmatched_groups():
    """
    If a list of non-identical group name lists is given, an error is raised.
    """
    group_cols = [["geography", "indicator"], ["geography", "mismatch"]]

    with pytest.raises(AssertionError):
        iup.extract_group_names(group_cols)


def test_extract_group_names_handles_no_groups():
    """
    If a list group name lists containing None is given, an error is raised.
    """
    group_cols = [[None], ["geography", "mismatch"]]

    with pytest.raises(AssertionError):
        iup.extract_group_names(group_cols)


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
