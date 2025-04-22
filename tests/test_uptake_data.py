import datetime as dt

import polars as pl
import pytest

import iup


def test_split_train_test(frame):
    """
    Return the data in two halves
    """
    frame2 = frame.with_columns(time_end=pl.col("time_end") + pl.duration(days=365))
    split_date = dt.date(2020, 6, 1)

    output = iup.UptakeData.split_train_test(
        iup.CumulativeUptakeData(pl.concat([frame, frame2])), split_date
    )

    assert output[0].equals(iup.CumulativeUptakeData(frame))

    assert output[1].equals(iup.CumulativeUptakeData(frame2))


def test_to_cumulative_handles_no_last(frame):
    """
    If last_cumulative is not given, then simple cumulative sums are performed
    """
    frame = iup.IncidentUptakeData(frame.drop(["N_vax", "N_tot"]))

    output = frame.to_cumulative(groups=["geography", "season"])

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.001,
                0.001,
                0.101,
                0.011,
                0.401,
                0.041,
                0.801,
                0.081,
            ]
        )
    )


def test_to_cumulative_handles_last(frame):
    """
    If last_cumulative is given, then cumulative sums are augmented
    """
    frame = iup.IncidentUptakeData(frame)

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
                0.002,
                0.111,
                0.012,
                0.411,
                0.042,
                0.811,
                0.082,
            ]
        )
    )


def test_to_cumulative_handles_no_groups(frame):
    """
    If there are no groups, cumulative sums are taken over the whole frame at once.
    Note that season is still considered a group, but there is only one unique season.
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("geography") == "USA").drop(["geography", "N_vax", "N_tot"])
    )

    output = frame.to_cumulative(groups=None)

    assert all(output["estimate"].round(10) == pl.Series([0.001, 0.101, 0.401, 0.801]))


def test_cumulative_uptake_is_proportion(frame):
    # should have an error if cumulative uptake is >1
    frame = frame.with_columns(estimate=pl.col("estimate") + 1.0)
    assert frame["estimate"].max() > 1.0
    with pytest.raises(AssertionError, match="proportion"):
        iup.CumulativeUptakeData(frame)

    # should not have an error if not
    iup.CumulativeUptakeData(frame.filter(pl.col("estimate") <= 1.0))


def test_to_incident_handles_groups(frame):
    """
    If there are groups, successive differences are taken over the groups.
    """
    output = frame.to_incident(groups=["geography", "season"])

    assert all(
        output["estimate"].round(10)
        == pl.Series([0.0, 0.0, 0.099, 0.009, 0.2, 0.02, 0.1, 0.01])
    )


def test_to_incident_handles_no_groups(frame):
    """
    If there are no groups, successive differences are taken over the entire data frame.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = frame.to_incident(groups=None)

    assert all(output["estimate"].round(10) == pl.Series([0.0, 0.099, 0.2, 0.1]))


def test_quantile_forecast_validation():
    with pytest.raises(AssertionError, match="quantile"):
        iup.QuantileForecast(
            {"quantile": [-0.1], "time_end": [dt.date(2020, 1, 1)], "estimate": [0.0]}
        )


def test_sample_forecast_validation():
    iup.SampleForecast(
        pl.DataFrame(
            {"time_end": [dt.date(2020, 1, 1)], "estimate": [0.0], "sample_id": "0"}
        ).with_columns(pl.col("sample_id"))
    )
