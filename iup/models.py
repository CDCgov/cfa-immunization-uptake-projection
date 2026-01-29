import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

import abc
import datetime
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from typing_extensions import Self

import iup


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
        groups: list[str,],
        model_params: dict[str, Any],
        mcmc_params: dict[str, Any],
        seed: int,
        date_column: str = "time_end",
    ):
        """Initialize with a seed and the model structure.

        Args:
            data: Cumulative coverage data for fitting and prediction.
            forecast_date: Date to split fit and prediction data.
            groups: Names of the columns of grouping factors, or `None` for no grouping.
            model_params: Parameter names and values to specify prior distributions.
            mcmc_params: Control parameters for MCMC fitting.
            seed: Random seed for stochastic elements of the model, to be split
                for fitting vs. predicting.
            date_column: Name of the date column in the data. Defaults to "time_end".
        """
        self.raw_data = data
        self.date_column = date_column
        self.forecast_date = forecast_date
        self.groups = groups
        self.model_params = model_params
        self.mcmc_params = mcmc_params
        self.fit_key, self.pred_key = random.split(random.key(seed), 2)

        # input validation
        assert "season" in self.groups
        assert {self.date_column, "elapsed", "N_vax", "N_tot"}.issubset(
            self.raw_data.columns
        )
        assert set(self.groups).issubset(self.raw_data.columns)

        # do the indexing
        self.n_group_levels = [
            self.raw_data.select(pl.col(group).unique()).height for group in self.groups
        ]
        self.data = self._index(self.raw_data, self.groups)

        # initialize MCMC. `None` is a placeholder indicating fitting has not occurred
        self.mcmc = None

    @staticmethod
    def _index(data: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
        """
        For each column in `groups` (e.g., `"season"`), add a new column `"{group}_idx"`
        (e.g., `"season_idx"`) that has the values in the original column replaced by
        integer indices.

        Args:
            data: dataframe
            groups: names of columns

        Returns: dataframe with additional columns like `"{group}_idx"`
        """
        for group in groups:
            unique_values = (
                data.select(pl.col(group).unique().sort()).get_column(group).to_list()
            )
            indices = list(range(len(unique_values)))
            replace_map = {value: index for value, index in zip(unique_values, indices)}
            data = data.with_columns(
                pl.col(group).replace_strict(replace_map).alias(f"{group}_idx")
            )

        return data

    def model(self, data: pl.DataFrame):
        if "N_vax" in data.columns:
            N_vax = jnp.array(data["N_vax"])
        else:
            N_vax = None

        return self._logistic_plus_linear(
            N_vax=N_vax,
            elapsed=jnp.array(data["elapsed"]),
            # jax runs into a problem if you don't specify this type
            N_tot=jnp.array(data["N_tot"], dtype=jnp.int32),
            groups=jnp.array(data.select([f"{group}_idx" for group in self.groups])),
            n_groups=len(self.groups),
            n_group_levels=self.n_group_levels,
            **self.model_params,
        )

    @staticmethod
    def _logistic_plus_linear(
        N_vax: jnp.ndarray | None,
        elapsed: jnp.ndarray,
        N_tot: jnp.ndarray,
        groups: jnp.ndarray,
        n_groups: int,
        n_group_levels: list[int],
        muA_shape1: float,
        muA_shape2: float,
        sigmaA_rate: float,
        tau_shape1: float,
        tau_shape2: float,
        K_shape: float,
        K_rate: float,
        muM_shape: float,
        muM_rate: float,
        sigmaM_rate: float,
        D_shape: float,
        D_rate: float,
    ):
        """Fit a mixed Logistic Plus Linear model on training data.

        Args:
            elapsed: Fraction of a year elapsed since the start of season at each data point.
            N_vax: Number of people vaccinated at each data point, or `None`.
            N_tot: Total number of people in the population at each data point.
            groups: Numeric codes for groups: row = data point, col = grouping factor.
            n_groups: Number of grouping factors.
            n_group_levels: Number of unique levels of each grouping factor.
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

        sigmaA = numpyro.sample(
            "sigmaA", dist.Exponential(sigmaA_rate), sample_shape=(n_groups,)
        )
        sigmaM = numpyro.sample(
            "sigmaM", dist.Exponential(sigmaM_rate), sample_shape=(n_groups,)
        )
        zA = numpyro.sample(
            "zA", dist.Normal(0, 1), sample_shape=(sum(n_group_levels),)
        )
        zM = numpyro.sample(
            "zM", dist.Normal(0, 1), sample_shape=(sum(n_group_levels),)
        )
        deltaA = zA * np.repeat(sigmaA, np.array(n_group_levels))
        deltaM = zM * np.repeat(sigmaM, np.array(n_group_levels))

        A = muA + np.sum(deltaA[groups], axis=1)
        M = muM + np.sum(deltaM[groups], axis=1)

        # Calculate latent true coverage at each datum
        v = A / (1 + jnp.exp(-K * (elapsed - tau))) + (M * elapsed)  # type: ignore

        numpyro.sample("obs", dist.BetaBinomial(v * D, (1 - v) * D, N_tot), obs=N_vax)  # type: ignore

    def fit(self) -> Self:
        """Fit a mixed Logistic Plus Linear model on training data.

        If grouping factors are specified, a hierarchical model will be built with
        group-specific parameters for the logistic maximum and linear slope,
        drawn from a shared distribution. Other parameters are non-hierarchical.

        Uses the data, groups, model_params, and mcmc_params specified during
        initialization.

        Returns:
            Self with the fitted model stored in the mcmc attribute.
        """
        self.kernel = NUTS(self.model, init_strategy=init_to_sample)
        self.mcmc = MCMC(self.kernel, **self.mcmc_params)
        self.mcmc.run(
            self.fit_key,
            self.data.filter(pl.col(self.date_column) <= self.forecast_date),
        )

        if "progress_bar" in self.mcmc_params and self.mcmc_params["progress_bar"]:
            self.mcmc.print_summary()

        return self

    def predict(self) -> pl.DataFrame:
        """Make projections from a fit Logistic Plus Linear model.

        Returns:
            Sample forecast data frame with predictions for dates after forecast_date.
        """

        assert self.mcmc is not None, f"Need to fit() first; mcmc is {self.mcmc}"

        predictive = Predictive(self.model, self.mcmc.get_samples())
        # run the predictions, not using the observations
        pred = predictive(self.pred_key, self.data.drop("N_vax"))

        # observations are rows; posterior samples are columns
        pred = np.array(pred["obs"]).transpose()

        # put predictions into a dataframe
        sample_cols = [f"_sample_{i}" for i in range(pred.shape[1])]
        pred = pl.DataFrame(pred, schema=sample_cols)

        index_cols = [self.date_column, "elapsed", "N_tot"] + self.groups

        # combine predictions
        return iup.SampleForecast(
            pl.concat([self.data, pred], how="horizontal")
            .unpivot(
                on=sample_cols,
                index=index_cols,
                variable_name="sample_id",
                value_name="estimate",
            )
            .with_columns(
                forecast_date=self.forecast_date,
                # convert from sample_id strings to integers
                sample_id=pl.col("sample_id")
                .replace_strict({name: i for i, name in enumerate(sample_cols)})
                .cast(pl.UInt64),
                estimate=pl.col("estimate") / pl.col("N_tot"),
            )
            .drop(["elapsed", "N_tot"])
        )
