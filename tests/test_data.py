import datetime as dt

import polars as pl
import pytest

import iup


def test_to_cumulative_handles_no_last(frame):
    """
    If last_cumulative is not given, then simple cumulative sums are performed
    """
    frame = iup.IncidentCoverageData(frame)

    output = frame.to_cumulative(groups=["geography", "season"])

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.001,
                0.003,
                0.006,
                0.01,
                0.015,
                0.021,
                0.028,
                0.036,
                0.045,
                0.055,
                0.001,
                0.003,
                0.006,
                0.01,
                0.015,
                0.021,
                0.028,
                0.036,
                0.045,
                0.055,
            ]
        )
    )


def test_to_cumulative_handles_last(frame):
    """
    If last_cumulative is given, then cumulative sums are augmented
    """
    frame = iup.IncidentCoverageData(frame)

    last_cumulative = pl.DataFrame(
        {
            "last_cumulative": [0.01, 0.001],
            "geography": ["USA", "PA"],
            "season": ["2019/2020"] * 2,
        }
    )

    output = frame.to_cumulative(
        groups=["geography", "season"],
        prev_cumulative=last_cumulative,
    )

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.011,
                0.013,
                0.016,
                0.02,
                0.025,
                0.031,
                0.038,
                0.046,
                0.055,
                0.065,
            ]
        )
    )


def test_to_cumulative_handles_no_groups(frame):
    """
    If there are no groups, cumulative sums are taken over the whole frame at once.
    Note that season is still considered a group, but there is only one unique season.
    """
    frame = iup.IncidentCoverageData(
        frame.filter(pl.col("geography") == "USA").drop(["geography"])
    )

    output = frame.to_cumulative(groups=None)

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.001,
                0.003,
                0.006,
                0.01,
                0.015,
                0.021,
                0.028,
                0.036,
                0.045,
                0.055,
                0.001,
                0.003,
                0.006,
                0.01,
                0.015,
                0.021,
                0.028,
                0.036,
                0.045,
                0.055,
            ]
        )
    )


def test_cumulative_coverage_is_proportion(frame):
    # should have an error if cumulative coverage is >1
    frame = frame.with_columns(estimate=pl.col("estimate") + 1.0)
    assert frame["estimate"].max() > 1.0
    with pytest.raises(AssertionError, match="proportion"):
        iup.CumulativeCoverageData(frame)

    # should not have an error if not
    iup.CumulativeCoverageData(frame.filter(pl.col("estimate") <= 1.0))


def test_to_incident_handles_groups(frame):
    """
    If there are groups, successive differences are taken over the groups.
    """
    output = frame.to_incident(groups=["geography", "season"])

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.0,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.0,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ]
        )
    )


def test_to_incident_handles_no_groups(frame):
    """
    If there are no groups, successive differences are taken over the entire data frame.
    """
    frame = iup.CumulativeCoverageData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = frame.to_incident(groups=None)

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.0,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.0,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ]
        )
    )


def test_quantile_forecast_validation():
    with pytest.raises(AssertionError, match="quantile"):
        iup.QuantileForecast(
            {"quantile": [-0.1], "time_end": [dt.date(2020, 1, 1)], "estimate": [0.0]}
        )


def test_sample_forecast_validation():
    iup.SampleForecast(
        pl.DataFrame(
            {"time_end": [dt.date(2020, 1, 1)], "estimate": [0.0], "sample_id": 0}
        ).with_columns(pl.col("sample_id").cast(pl.UInt64))
    )
