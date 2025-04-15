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
            "N_vax": [1, 1, 100, 10, 300, 30, 400, 40],
            "N_tot": [1000] * 8,
            "estimate": [0.001, 0.001, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04],
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
        "A_shape1": 100.0,
        "A_shape2": 140.0,
        "A_sig": 40.0,
        "H_shape1": 100.0,
        "H_shape2": 225.0,
        "n_shape": 20.0,
        "n_rate": 5.0,
        "M_shape": 1.0,
        "M_rate": 0.1,
        "M_sig": 40,
        "d_shape": 5.0,
        "d_rate": 0.01,
    }

    return params


@pytest.fixture
def mcmc_params():
    """
    Mock set of mcmc control parameters.
    """
    mcmc = {"num_warmup": 10, "num_samples": 10, "num_chains": 1}

    return mcmc


def test_augment_data(frame):
    """
    Add a column for time elapsed since season start
    """
    output = iup.models.LLModel.augment_data(frame, 9, 1)

    assert [round(i * 365, 1) for i in output["elapsed"].to_list()] == [
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
    data = iup.models.LLModel.augment_data(frame, 9, 1)
    iup.models.LLModel(0).fit(data, None, params, mcmc_params)


def test_fit_handles_groups(frame, params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )
    data = iup.models.LLModel.augment_data(frame, 9, 1)
    model = iup.models.LLModel(0).fit(data, ["season"], params, mcmc_params)

    dimensions = [value.shape[0] for key, value in model.mcmc.get_samples().items()]

    assert all(d == 10 for d in dimensions)


def test_augment_scaffold(frame):
    """
    Add elapsed an elapsed column to a scaffold.
    """
    output = iup.models.LLModel.augment_scaffold(frame, 9, 1)

    assert output.shape[1] == frame.shape[1]
    assert [round(i * 365, 1) for i in output["elapsed"].to_list()] == [
        121.0,
        121.0,
        128.0,
        128.0,
        135.0,
        135.0,
        142.0,
        142.0,
    ]
