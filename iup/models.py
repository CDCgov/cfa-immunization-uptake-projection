import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

import abc
import datetime
from typing import Any, List

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from typing_extensions import Self

import iup
import iup.utils


class CoverageModel(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        data: pl.DataFrame,
        forecast_date: datetime.date,
        groups: list[str] | None,
        model_params: dict[str, Any],
        mcmc_params: dict[str, Any],
        seed: int,
    ):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self) -> pl.DataFrame:
        pass


class LPLModel(CoverageModel):
    """
    Subclass of CoverageModel for a mixed Logistic Plus Linear model.
    For details, see the online docs.
    """

    def __init__(
        self,
        data: iup.CumulativeCoverageData,
        forecast_date: datetime.date,
        groups: list[str,] | None,
        model_params: dict[str, Any],
        mcmc_params: dict[str, Any],
        seed: int,
        date_column: str = "time_end",
    ):
        """Initialize with a seed and the model structure.

        Args:
            seed: Random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.data = data
        self.date_column = date_column
        self.forecast_date = forecast_date
        self.groups = groups
        self.model_params = model_params
        self.mcmc_params = mcmc_params
        self.fit_key, self.pred_key = random.split(random.key(seed), 2)

        # check that the data have the expected columns
        assert {self.date_column, "elapsed"}.issubset(self.data.columns)

        # do the indexing
        self.group_combos = extract_group_combos(self.data, self.groups)

        if self.groups is not None:
            assert set(self.groups).issubset(self.data.columns)

            self.num_group_factors = len(self.groups)
            self.num_group_levels = iup.utils.count_unique_values(self.group_combos)
            self.value_to_index = iup.utils.map_value_to_index(data.select(self.groups))
            self.group_codes = iup.utils.value_to_index(
                data.select(groups), self.value_to_index, self.num_group_levels
            )
        else:
            self.group_codes = None
            self.num_group_factors = 0
            self.num_group_levels = [0]
            self.value_to_index = None

        # split fit and prediction data
        self.fit_data = self.data.filter(pl.col(self.date_column) <= self.forecast_date)
        self.pred_data = self.data.filter(pl.col(self.date_column) > self.forecast_date)

        self.mcmc = None

    @staticmethod
    def _logistic_plus_linear(
        elapsed: np.ndarray,
        N_tot: int,
        N_vax: np.ndarray | None = None,
        groups=None,
        num_group_factors=0,
        num_group_levels=[0],
        muA_shape1=100.0,
        muA_shape2=180.0,
        sigmaA_rate=40.0,
        tau_shape1=100.0,
        tau_shape2=225.0,
        K_shape=25.0,
        K_rate=1.0,
        muM_shape=1.0,
        muM_rate=10.0,
        sigmaM_rate=40.0,
        D_shape=350.0,
        D_rate=1.0,
    ):
        """Fit a mixed Logistic Plus Linear model on training data.

        Args:
            elapsed: Fraction of a year elapsed since the start of season at each data point.
            N_vax: Number of people vaccinated at each data point.
            N_tot: Number of people contacted at each data point.
            groups: Numeric codes for groups: row = data point, col = grouping factor.
            num_group_factors: Number of grouping factors.
            num_group_levels: Number of unique levels of each grouping factor.
            muA_shape1: Beta distribution shape1 parameter for muA prior.
            muA_shape2: Beta distribution shape2 parameter for muA prior.
            sigmaA_rate: Exponential distribution rate parameter for sigmaA prior.
            tau_shape1: Beta distribution shape1 parameter for tau prior.
            tau_shape2: Beta distribution shape2 parameter for tau prior.
            K_shape: Gamma distribution shape parameter for K prior.
            K_rate: Gamma distribution rate parameter for K prior.
            muM_shape: Gamma distribution shape parameter for muM prior.
            muM_rate: Gamma distribution rate parameter for muM prior.
            sigmaM_rate: Exponential distribution rate parameter for sigmaM prior.
            D_shape: Gamma distribution shape parameter for D prior.
            D_rate: Gamma distribution rate parameter for D prior.
        """
        # Sample the overall average value for each parameter
        muA = numpyro.sample("muA", dist.Beta(muA_shape1, muA_shape2))
        muM = numpyro.sample("muM", dist.Gamma(muM_shape, muM_rate))
        tau = numpyro.sample("tau", dist.Beta(tau_shape1, tau_shape2))
        K = numpyro.sample("K", dist.Gamma(K_shape, K_rate))
        D = numpyro.sample("d", dist.Gamma(D_shape, D_rate))

        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            sigmaA = numpyro.sample(
                "sigmaA",
                dist.Exponential(sigmaA_rate),
                sample_shape=(num_group_factors,),
            )
            sigmaM = numpyro.sample(
                "sigmaM",
                dist.Exponential(sigmaM_rate),
                sample_shape=(num_group_factors,),
            )
            zA = numpyro.sample(
                "zA", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            )
            zM = numpyro.sample(
                "zM", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            )
            deltaA = zA * np.repeat(sigmaA, np.array(num_group_levels))
            deltaM = zM * np.repeat(
                sigmaM,
                np.array(
                    num_group_levels,
                ),
            )

            A = muA + np.sum(deltaA[groups], axis=1)
            M = muM + np.sum(deltaM[groups], axis=1)
        else:
            A = muA
            M = muM

        # Calculate latent true coverage at each datum
        v = A / (1 + jnp.exp(-K * (elapsed - tau))) + (M * elapsed)  # type: ignore

        numpyro.sample("obs", dist.BetaBinomial(v * D, (1 - v) * D, N_tot), obs=N_vax)  # type: ignore

    def fit(self) -> Self:
        """Fit a mixed Logistic Plus Linear model on training data.

        If grouping factors are specified, a hierarchical model will be built with
        group-specific parameters for the logistic maximum and linear slope,
        drawn from a shared distribution. Other parameters are non-hierarchical.

        Args:
            data: Training data on which to fit the model.
            groups: Names of the columns for the grouping factors.
            params: Parameter names and values to specify prior distributions.
            mcmc: Control parameters for MCMC fitting.

        Returns:
            Model object with grouping factor combinations and the model fit stored as attributes.
        """
        self.kernel = NUTS(self._logistic_plus_linear, init_strategy=init_to_sample)
        self.mcmc = MCMC(self.kernel, **self.mcmc_params)

        self.mcmc.run(
            self.fit_key,
            elapsed=self.fit_data["elapsed"].to_numpy(),
            N_vax=self.fit_data["N_vax"].to_numpy(),
            groups=self.group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            **self.model_params,
        )

        if "progress_bar" in self.mcmc_params and self.mcmc_params["progress_bar"]:
            self.mcmc.print_summary()

        return self

    def predict(self) -> pl.DataFrame:
        """Make projections from a fit Logistic Plus Linear model"""

        assert self.mcmc is not None, "Need to fit() first"

        predictive = Predictive(self._logistic_plus_linear, self.mcmc.get_samples())

        predictions = np.array(
            predictive(
                self.pred_key,
                elapsed=self.pred_data["elapsed"].to_numpy(),
                groups=self.group_codes,
                num_group_factors=self.num_group_factors,
                num_group_levels=self.num_group_levels,
            )["obs"]
        ).transpose()

        if self.groups is None:
            index_cols = [self.date_column, "season"]
        elif "season" in self.groups:
            index_cols = [self.date_column] + self.groups
        else:
            index_cols = [self.date_column, "season"] + self.groups

        pred = (
            pl.concat(
                [
                    self.pred_data,
                    pl.DataFrame(
                        predictions,
                        schema=[f"{i + 1}" for i in range(predictions.shape[1])],
                    ),
                ],
                how="horizontal",
            )
            .unpivot(
                index=index_cols,
                variable_name="sample_id",
                value_name="estimate",
            )
            .with_columns(
                forecast_date=self.forecast_date,
                sample_id=pl.col("sample_id"),
                estimate=pl.col("estimate") / pl.col("N_tot"),
            )
            .drop(["elapsed", "N_tot"])
        )

        return iup.SampleForecast(pred)


def extract_group_combos(
    data: pl.DataFrame, groups: List[str,] | None
) -> pl.DataFrame | None:
    """Extract from coverage data all combinations of grouping factors.

    Args:
        data: Coverage data possibly containing grouping factors.
        groups: Names of the columns for the grouping factors.

    Returns:
        All combinations of grouping factors.
    """
    if groups is not None:
        return data.select(groups).unique()
    else:
        return None
