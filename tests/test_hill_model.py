import polars as pl
import pytest

import iup
import iup.models


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = pl.DataFrame(
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
        }
    ).with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))

    frame = iup.IncidentUptakeData(frame)

    return frame


@pytest.fixture
def params():
    """
    Mock set of parameter values to specify the LIUM prior distributions.
    """

    params = {
        "n_low": 1.0,
        "n_high": 5.0,
        "A_low": 0.0,
        "A_high": 1.0,
        "A_sig": 1.0,
        "H_low": 10.0,
        "H_high": 180.0,
        "H_sig": 1.0,
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


def test_augment_data(frame):  # LEFT OFF HERE
    """
    Add a column for time elapsed since season start
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = iup.IncidentUptakeData(
        frame.drop(["elapsed", "interval", "previous", "daily"])
    )

    output = iup.models.LinearIncidentUptakeModel.augment_columns(output, None)

    assert frame.equals(output)


def test_augment_columns_handles_groups(frame):
    """
    Add columns for elapsed, interval, previous, and daily
    """
    output = iup.IncidentUptakeData(
        frame.drop(["elapsed", "interval", "previous", "daily"])
    )

    output = iup.models.LinearIncidentUptakeModel.augment_columns(
        output, ["geography", "season"]
    )

    assert frame.equals(output)


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


def test_augment_scaffold_handles_no_groups(frame):
    """
    Add elapsed and interval columns to a scaffold
    """
    frame = (
        frame.filter(pl.col("geography") == "USA")
        .drop(["geography", "elapsed", "interval", "daily", "previous"])
        .with_columns(estimate=0.0)
    )

    start = pl.DataFrame({"last_elapsed": 100.0, "last_interval": 7.0})

    scaffold = iup.models.LinearIncidentUptakeModel.augment_scaffold(frame, None, start)

    output = pl.concat(
        [
            frame.drop("estimate"),
            pl.DataFrame(
                {"elapsed": [107.0, 114.0, 121.0, 128.0], "interval": [7.0] * 4}
            ),
        ],
        how="horizontal",
    )

    assert output.equals(scaffold)


def test_augment_scaffold_handles_groups(frame):
    """
    Add elapsed and interval columns to a scaffold
    """
    frame = frame.drop(["elapsed", "interval", "daily", "previous"]).with_columns(
        estimate=0.0
    )

    start = pl.DataFrame(
        {
            "last_elapsed": [100.0, 100.0],
            "last_interval": [7.0, 7.0],
            "geography": ["USA", "PA"],
            "season": ["2019/2020", "2019/2020"],
        }
    )

    scaffold = iup.models.LinearIncidentUptakeModel.augment_scaffold(
        frame, ["geography", "season"], start
    )

    output = pl.concat(
        [
            frame.drop("estimate"),
            pl.DataFrame(
                {
                    "elapsed": [107.0, 107.0, 114.0, 114.0, 121.0, 121.0, 128.0, 128.0],
                    "interval": [7.0] * 8,
                }
            ),
        ],
        how="horizontal",
    )

    assert output.equals(scaffold)
