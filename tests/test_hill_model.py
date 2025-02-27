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
            "estimate": [0.0, 0.0, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04],
            "season": "2019/2020",
        }
    ).with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))

    frame = iup.CumulativeUptakeData(frame)

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
    output = iup.models.HillModel.augment_data(frame, 9, 1, None, None)

    assert output["elapsed"].to_list() == [
        121.0,
        121.0,
        128.0,
        128.0,
        135.0,
        135.0,
        142.0,
        142.0,
    ]


def test_fit_handles_no_groups(frame, params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )
    data = iup.models.HillModel.augment_data(frame, 9, 1, None, None)
    model = iup.models.HillModel(0).fit(data, None, params, mcmc_params)

    dimensions = [value.shape[0] for key, value in model.mcmc.get_samples().items()]

    assert all(d == 10 for d in dimensions)


def test_fit_handles_groups(frame, params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )
    data = iup.models.HillModel.augment_data(frame, 9, 1, None, None)
    model = iup.models.HillModel(0).fit(data, ["season"], params, mcmc_params)

    dimensions = [value.shape[0] for key, value in model.mcmc.get_samples().items()]

    assert all(d == 10 for d in dimensions)


def test_date_to_elapsed(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.models.HillModel.date_to_elapsed(pl.col("time_end"), 9, 1).over(
            "geography"
        )
    )

    expected = pl.Series(
        [
            121.0,
            121.0,
            128.0,
            128.0,
            135.0,
            135.0,
            142.0,
            142.0,
        ]
    )

    assert (output["elapsed"] == expected).all()


def test_augment_scaffold(frame):
    """
    Add elapsed an elapsed column to a scaffold.
    """
    output = iup.models.HillModel.augment_scaffold(frame, 9, 1)

    assert output.shape[1] == frame.shape[1]
    assert output["elapsed"].to_list() == [
        121.0,
        121.0,
        128.0,
        128.0,
        135.0,
        135.0,
        142.0,
        142.0,
    ]
