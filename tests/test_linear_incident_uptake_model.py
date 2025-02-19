import datetime as dt

import polars as pl
import polars.testing
import pytest

import iup
import iup.models


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = (
        pl.DataFrame(
            {
                "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
                "time_end": [
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
        .with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))
    )

    frame = iup.IncidentUptakeData(frame)

    return frame


@pytest.fixture
def params():
    """
    Mock set of parameter values to specify the LIUM prior distributions.
    """

    params = {
        "a_mn": 0.0,
        "a_sd": 1.0,
        "bP_mn": 0.0,
        "bP_sd": 1.0,
        "bE_mn": 0.0,
        "bE_sd": 1.0,
        "bPE_mn": 0.0,
        "bPE_sd": 1.0,
        "sig_mn": 1.0,
    }

    return params


@pytest.fixture
def mcmc_params():
    """
    Mock set of mcmc control parameters.
    """
    mcmc = {"num_warmup": 10, "num_samples": 10, "num_chains": 1}

    return mcmc


def test_extract_starting_conditions(frame):
    """
    Use the last date for each grouping factor
    """
    output = iup.models.LinearIncidentUptakeModel.extract_starting_conditions(
        frame,
        [
            "geography",
            "season",
        ],
    )

    assert output.sort("geography").equals(
        pl.DataFrame(
            {
                "geography": ["USA", "PA"],
                "season": ["2019/2020", "2019/2020"],
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
    output = iup.models.LinearIncidentUptakeModel.extract_standards(
        frame, ("estimate", "elapsed")
    )

    correct = {
        "estimate": {"mean": frame["estimate"].mean(), "std": frame["estimate"].std()},
        "elapsed": {"mean": frame["elapsed"].mean(), "std": frame["elapsed"].std()},
    }

    assert output == correct


def test_fit(frame, params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """

    data = iup.IncidentUptakeData(frame)
    model = iup.models.LinearIncidentUptakeModel(0).fit(
        data, ["geography", "season"], params, mcmc_params
    )

    dimensions = [value.shape[0] for key, value in model.mcmc.get_samples().items()]

    assert all(d == 10 for d in dimensions)


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
    ).join(
        pl.DataFrame(
            {"geography": ["USA", "PA"], "season": ["2019/2020", "2019/2020"]}
        ),
        how="cross",
    )
    start_date = dt.date(2020, 2, 1)
    end_date = dt.date(2020, 2, 29)
    interval = "7d"
    group_cols = ["geography", "season"]

    output = iup.models.LinearIncidentUptakeModel.build_scaffold(
        start, start_date, end_date, interval, group_cols
    )

    assert output.shape == (10, 8)


def test_trim_outlier_intervals_handles_two_rows(frame):
    """
    If there are two or fewer rows (per group), all rows should be trimmed.
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("time_end") < dt.date(2020, 1, 9))
    )

    output = iup.models.LinearIncidentUptakeModel.trim_outlier_intervals(
        frame,
        group_cols=["geography"],
    )

    assert output.shape[0] == 0


def test_trim_outlier_intervals_handles_above_threshold():
    """
    If the first interval is too big, first three rows are trimmed by group.
    """
    df = (
        pl.DataFrame(
            {
                "geography": ["USA"] * 4 + ["PA"] * 4,
                "time_end": [
                    dt.date(2019, 12, 31),
                    dt.date(2020, 1, 7),
                    dt.date(2020, 1, 14),
                    dt.date(2020, 1, 21),
                ]
                * 2,
                "estimate": [0.0, 1.0, 3.0, 4.0] * 2,
            }
        )
        # shuffle the rows, to check that grouping & sorting work regardless
        .sample(seed=1234)
        # validate
        .pipe(iup.IncidentUptakeData)
    )

    output = iup.models.LinearIncidentUptakeModel.trim_outlier_intervals(
        df,
        group_cols=["geography"],
    )

    # should drop the first 3 rows, leaving only Jan 21
    expected = output.filter(pl.col("time_end") == dt.date(2020, 1, 21))

    polars.testing.assert_frame_equal(output, expected, check_row_order=False)


def test_trim_outlier_intervals_handles_below_threshold(frame):
    """
    If the first interval is not too big, first two rows are trimmed by group.
    """
    frame = iup.IncidentUptakeData(frame)

    output = iup.models.LinearIncidentUptakeModel.trim_outlier_intervals(
        frame,
        group_cols=["geography"],
        threshold=2,
    )

    assert output.shape[0] == 4


def test_trim_outlier_intervals_handles_zero_std(frame):
    """
    If std dev of intervals is 0, first two rows are trimmed by group
    """
    frame = frame.filter(pl.col("time_end") > dt.date(2020, 1, 1))
    frame = iup.IncidentUptakeData(frame)

    output = iup.models.LinearIncidentUptakeModel.trim_outlier_intervals(
        frame,
        group_cols=["geography"],
    )

    assert output.shape[0] == 2


def test_augment_implicit_columns(frame):
    """
    Add 5 columns to the incident uptake data without losing any rows
    """
    frame = iup.IncidentUptakeData(frame)
    frame = iup.models.LinearIncidentUptakeModel.augment_implicit_columns(
        frame,
        group_cols=["geography"],
    )

    assert frame.shape[0] == 8
    assert frame.shape[1] == 8


def test_date_to_season(frame):
    """
    Return the overwinter season, for both fall and spring dates
    """
    df = pl.DataFrame({"date": [dt.date(2020, 1, 1), dt.date(2020, 12, 31)]})
    current = df.select(iup.UptakeData.date_to_season(pl.col("date"))).to_series()
    expected = pl.Series("date", ["2019/2020", "2020/2021"])

    assert (current == expected).all()


def test_date_to_interval(frame):
    """
    Return the interval between dates by grouping factor
    """
    output = frame.sort(["geography", "time_end"]).with_columns(
        interval=iup.models.LinearIncidentUptakeModel.date_to_interval(
            pl.col("time_end")
        ).over("geography")
    )

    expected = pl.Series("interval", ([None] + [7.0] * 3) * 2)

    assert (output["interval"] == expected).all()


def test_date_to_elapsed(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.models.LinearIncidentUptakeModel.date_to_elapsed(
            pl.col("time_end")
        ).over("geography")
    )

    expected = pl.Series([0.0, 0.0, 7.0, 7.0, 14.0, 14.0, 21.0, 21.0])

    assert (output["elapsed"] == expected).all()
