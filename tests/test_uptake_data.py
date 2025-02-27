import datetime as dt

import polars as pl
import pytest

import iup


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


def test_split_train_test_handles_train(frame):
    """
    Return the training half of a data set.
    """
    frame2 = frame.with_columns(time_end=pl.col("time_end") + pl.duration(days=365))
    start_date = dt.date(2020, 6, 1)

    output = iup.UptakeData.split_train_test(
        iup.CumulativeUptakeData(pl.concat([frame, frame2])), start_date, "train"
    )

    assert output.equals(iup.CumulativeUptakeData(frame))


def test_split_train_test_handles_test(frame):
    """
    Return the testing half of a data set.
    """
    frame2 = frame.with_columns(time_end=pl.col("time_end") + pl.duration(days=365))
    start_date = dt.date(2020, 6, 1)

    output = iup.UptakeData.split_train_test(
        iup.CumulativeUptakeData(pl.concat([frame, frame2])), start_date, "test"
    )

    assert output.equals(frame2)


def test_date_to_season(frame):
    """
    Season is extracted from a column of dates
    """
    frame = frame.with_columns(
        season2=pl.col("time_end").pipe(iup.UptakeData.date_to_season)
    )

    assert all(frame["season"] == frame["season2"])


def test_to_cumulative_handles_no_last(frame):
    """
    If last_cumulative is not given, then simple cumulative sums are performed
    """
    frame = iup.IncidentUptakeData(frame)

    output = frame.to_cumulative(groups=["geography", "season"])

    assert all(
        output["estimate"]
        == pl.Series(
            [
                0.0,
                0.0,
                0.01,
                0.001,
                0.04,
                0.004,
                0.08,
                0.008,
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
        last_cumulative=last_cumulative,
    )

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.01,
                0.001,
                0.02,
                0.002,
                0.05,
                0.005,
                0.09,
                0.009,
            ]
        )
    )


def test_to_cumulative_handles_no_groups(frame):
    """
    If there are no groups, cumulative sums are taken over the whole frame at once.
    Note that season is still considered a group, but there is only one unique season.
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = frame.to_cumulative(groups=None)

    assert all(output["estimate"] == pl.Series([0.0, 0.01, 0.04, 0.08]))


def test_trim_outlier_intervals_handles_two_rows(frame):
    """
    If there are two or fewer rows (per group), all rows should be trimmed.
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("time_end") < dt.date(2020, 1, 9)).with_columns(
            interval=pl.col("time_end").diff().dt.total_days().cast(pl.Float64)
        )
    )

    output = frame.trim_outlier_intervals(
        groups=["geography", "season"],
    )

    assert output.shape[0] == 0


def test_trim_outlier_intervals_handles_above_threshold():
    """
    If the first interval is too big, first three rows are trimmed by group.
    """
    df = iup.IncidentUptakeData(
        pl.DataFrame(
            {
                "geography": ["USA"] * 4 + ["PA"] * 4,
                "time_end": [
                    dt.date(2019, 12, 24),
                    dt.date(2020, 1, 7),
                    dt.date(2020, 1, 14),
                    dt.date(2020, 1, 21),
                ]
                * 2,
                "estimate": [0.0, 1.0, 3.0, 4.0] * 2,
                "season": ["2019/2020"] * 8,
                "interval": [None, 14, 7, 7, None, 14, 7, 7],
            }
        ).sort("time_end")
    )

    output = df.trim_outlier_intervals(
        groups=["geography", "season"],
    )

    assert output["time_end"].unique().to_list() == [dt.date(2020, 1, 21)]


def test_trim_outlier_intervals_handles_below_threshold(frame):
    """
    If the first interval is not too big, first two rows are trimmed by group.
    """
    frame = iup.IncidentUptakeData(
        frame.with_columns(
            interval=pl.col("time_end").diff().dt.total_days().cast(pl.Float64)
        )
    )

    output = frame.trim_outlier_intervals(
        groups=["geography", "season"],
        threshold=2,
    )

    assert output.shape[0] == 4


def test_trim_outlier_intervals_handles_zero_std(frame):
    """
    If std dev of intervals is 0, first two rows are trimmed by group
    """
    frame = frame.filter(pl.col("time_end") > dt.date(2020, 1, 1)).with_columns(
        interval=pl.col("time_end").diff().dt.total_days().cast(pl.Float64)
    )
    frame = iup.IncidentUptakeData(frame)

    output = frame.trim_outlier_intervals(
        groups=["geography", "season"],
    )

    assert output.shape[0] == 2


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
    frame = iup.CumulativeUptakeData(frame.filter(pl.col("estimate") <= 0.01))

    output = frame.to_incident(groups=["geography", "season"])

    assert all(
        output["estimate"].round(10) == pl.Series([0.0, 0.0, 0.01, 0.001, 0.002, 0.001])
    )


def test_to_incident_handles_no_groups(frame):
    """
    If there are no groups, successive differences are taken over the entire data frame.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA", pl.col("estimate") <= 0.01).drop(
            "geography"
        )
    )

    output = frame.to_incident(groups=None)

    assert all(output["estimate"].round(10) == pl.Series([0.0, 0.01]))


def test_insert_rollouts_handles_groups(frame):
    """
    If grouping columns are given to insert_rollouts, a separate rollout is inserted for each group.
    """
    rollouts = [dt.date(2019, 1, 1)]
    group_cols = ["geography", "season"]

    output = iup.CumulativeUptakeData(frame).insert_rollouts(rollouts, group_cols, 9, 1)

    assert output.shape[0] == 10
    assert (
        output["time_end"]
        .value_counts()
        .filter(pl.col("time_end") == rollouts[0])["count"][0]
        == 2
    )
    assert output["time_end"].is_sorted()


def test_insert_rollouts_handles_no_groups(frame):
    """
    If no grouping columns are given to insert_rollouts, only one of each rollout is inserted.
    """
    rollouts = [dt.date(2019, 1, 1)]
    group_cols = None
    output = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop(["geography"])
    ).insert_rollouts(rollouts, group_cols, 9, 1)

    assert output.shape[0] == 5
    assert (
        output["time_end"]
        .value_counts()
        .filter(pl.col("time_end") == rollouts[0])["count"][0]
        == 1
    )
    assert output["time_end"].is_sorted()


def test_quantile_forecast_validation():
    with pytest.raises(AssertionError, match="quantile"):
        iup.QuantileForecast(
            {"quantile": [-0.1], "time_end": [dt.date(2020, 1, 1)], "estimate": [0.0]}
        )


def test_sample_forecast_validation():
    iup.SampleForecast(
        pl.DataFrame(
            {"time_end": [dt.date(2020, 1, 1)], "estimate": [0.0], "sample_id": 0}
        ).with_columns(pl.col("sample_id").cast(pl.Int64))
    )
