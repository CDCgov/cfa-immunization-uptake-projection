import datetime

import numpyro.infer
import polars as pl
import pytest

import iup
import iup.models


@pytest.fixture
def model_params():
    """
    Mock set of parameter values to specify the LIUM prior distributions.
    """

    params = {
        "muA_shape1": 100.0,
        "muA_shape2": 140.0,
        "sigmaA_rate": 40.0,
        "tau_shape1": 100.0,
        "tau_shape2": 225.0,
        "K_shape": 20.0,
        "K_rate": 5.0,
        "muM_shape": 1.0,
        "muM_rate": 0.1,
        "sigmaM_rate": 40,
        "D_shape": 5.0,
        "D_rate": 0.01,
        "N_tot": 1000,
    }

    return params


@pytest.fixture
def mcmc_params():
    """
    Mock set of mcmc control parameters.
    """
    mcmc = {"num_warmup": 10, "num_samples": 10, "num_chains": 1}

    return mcmc


def test_fit_handles_no_groups(frame, model_params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """
    data = iup.CumulativeCoverageData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    model = iup.models.LPLModel(
        data=data,
        forecast_date=datetime.date(2020, 1, 21),
        groups=None,
        model_params=model_params,
        mcmc_params=mcmc_params,
        seed=0,
    )

    model.fit()


def test_fit_handles_groups(frame, model_params, mcmc_params):
    """
    Model should produce posterior samples for each parameter.
    """
    data = iup.CumulativeCoverageData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    model = iup.models.LPLModel(
        data=data,
        forecast_date=datetime.date(2020, 1, 21),
        groups=["season"],
        model_params=model_params,
        mcmc_params=mcmc_params,
        seed=0,
    )

    model.fit()
    assert isinstance(model.mcmc, numpyro.infer.MCMC)

    dimensions = [value.shape[0] for _, value in model.mcmc.get_samples().items()]
    assert all(d == 10 for d in dimensions)
