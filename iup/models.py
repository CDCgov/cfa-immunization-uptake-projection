import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

import abc
from typing import List

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from typing_extensions import Self

import iup.utils
from iup import CumulativeUptakeData, SampleForecast, UptakeData


class UptakeModel(abc.ABC):
    """
    Abstract class for different types of models.
    Every subclass of model will have some core methods of the same name.
    """

    @staticmethod
    @abc.abstractmethod
    def augment_data(
        data: UptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> UptakeData:
        """Add columns to preprocessed uptake data to provide all input information that a specific model requires.

        Args:
            data: Preprocessed uptake data.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            Augmented uptake data.
        """
        pass

    @abc.abstractmethod
    def fit(
        self,
        data: UptakeData,
        groups: List[str,] | None,
        params: dict,
        mcmc: dict,
    ) -> Self:
        """Fit a model on its properly augmented data.

        Args:
            data: Training data.
            groups: Names of columns for grouping factors.
            params: Parameter names and values to specify prior distributions.
            mcmc: Control parameters for MCMC fitting.

        Returns:
            Fitted model object.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def augment_scaffold(
        scaffold: pl.DataFrame,
        groups: List[str] | None,
        start: pl.DataFrame,
    ) -> pl.DataFrame:
        """Add columns to a scaffold of dates for forecasting to provide all input information that a specific model requires.

        Args:
            scaffold: Scaffold of dates for forecasting.
            groups: Names of columns for grouping factors.
            start: Starting values for forecasting.

        Returns:
            Augmented scaffold.
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        test_data: pl.DataFrame,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """Use a fit model to fill in forecasts in a scaffold of dates.

        Args:
            test_data: Test data scaffold.
            groups: Names of columns for grouping factors.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            Forecast data.
        """
        pass

    @abc.abstractmethod
    def __init__(self, seed: int):
        pass

    # save for future models #
    mcmc = None


class LPLModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Logistic Plus Linear model.
    For details, see the online docs.
    """

    def __init__(self, seed: int):
        """Initialize with a seed and the model structure.

        Args:
            seed: Random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.rng_key = random.key(seed)
        self.fit_key, self.pred_key = random.split(self.rng_key, 2)
        self.model = LPLModel._logistic_plus_linear

    @staticmethod
    def _logistic_plus_linear(
        elapsed,
        N_vax=None,
        N_tot=None,
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

        # Calculate latent true uptake at each datum
        v = A / (1 + jnp.exp(0 - K * (elapsed - tau))) + (M * elapsed)

        numpyro.sample("obs", dist.BetaBinomial(v * D, (1 - v) * D, N_tot), obs=N_vax)  # type: ignore

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """Format preprocessed data for fitting a Logistic Plus Linear model.

        Args:
            data: Training data for fitting a Logistic Plus Linear model.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            Cumulative uptake data ready for fitting a Logistic Plus Linear model.

        Note:
            The following steps are required to prepare preprocessed data
            for fitting a linear incident uptake model:
            - Add an extra column for time elapsed since start-of-season
            - Rescale this time elapsed to a proportion of the year
        """
        data = CumulativeUptakeData(
            data.with_columns(
                elapsed=iup.utils.date_to_elapsed(
                    pl.col("time_end"),
                    season_start_month,
                    season_start_day,
                )
                / 365
            )
        )

        return data

    def fit(
        self,
        data: CumulativeUptakeData,
        groups: List[str,] | None,
        params: dict,
        mcmc: dict,
    ) -> Self:
        """Fit a mixed Logistic Plus Linear model on training data.

        Args:
            data: Training data on which to fit the model.
            groups: Names of the columns for the grouping factors.
            params: Parameter names and values to specify prior distributions.
            mcmc: Control parameters for MCMC fitting.

        Returns:
            Model object with grouping factor combinations and the model fit stored as attributes.

        Note:
            If grouping factors are specified, a hierarchical model will be built with
            group-specific parameters for the logistic maximum and linear slope,
            drawn from a shared distribution. Other parameters are non-hierarchical.
        """
        self.group_combos = extract_group_combos(data, groups)

        # Tranform the levels of the grouping factors into numeric codes
        if groups is not None:
            self.num_group_factors = len(groups)
            self.num_group_levels = iup.utils.count_unique_values(self.group_combos)
            self.value_to_index = iup.utils.map_value_to_index(data.select(groups))
            group_codes = iup.utils.value_to_index(
                data.select(groups), self.value_to_index, self.num_group_levels
            )
        else:
            group_codes = None
            self.num_group_factors = 0
            self.num_group_levels = [0]
            self.value_to_index = None

        # Prepare the data to be fed to the model. Must be numpy arrays.
        elapsed = data["elapsed"].to_numpy()
        N_vax = data["N_vax"].to_numpy()
        N_tot = data["N_tot"].to_numpy()

        self.kernel = NUTS(self.model, init_strategy=init_to_sample)
        self.mcmc = MCMC(self.kernel, **mcmc)

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            muA_shape1=params["muA_shape1"],
            muA_shape2=params["muA_shape2"],
            sigmaA_rate=params["sigmaA_rate"],
            tau_shape1=params["tau_shape1"],
            tau_shape2=params["tau_shape2"],
            K_shape=params["K_shape"],
            K_rate=params["K_rate"],
            muM_shape=params["muM_shape"],
            muM_rate=params["muM_rate"],
            sigmaM_rate=params["sigmaM_rate"],
            D_shape=params["D_shape"],
            D_rate=params["D_rate"],
        )

        if "progress_bar" in mcmc and mcmc["progress_bar"]:
            self.mcmc.print_summary()

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame,
        season_start_month: int,
        season_start_day: int,
        N_tot: int = 10_000,
    ) -> pl.DataFrame:
        """Add columns to a scaffold of dates for forecasting from a Logistic Plus Linear model.

        Args:
            scaffold: Scaffold of dates for forecasting.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.
            N_tot: Predictions are made as if this many individuals are sampled.

        Returns:
            Scaffold with extra columns required by the Logistic Plus Linear model.
        """
        return scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=N_tot,
        )

    def predict(
        self,
        test_data: pl.DataFrame,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """Make projections from a fit Logistic Plus Linear model.

        Args:
            test_data: Exact target dates to use, when test data exists.
            groups: Names of the columns for the grouping factors.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            Model with incident and cumulative projections as attributes.

        Note:
            A data frame is set up to house the projections over the
            desired time window with the desired intervals.
            Forecasts are then made for each date in this scaffold.
        """
        assert "time_end" in test_data.columns

        if groups is None:
            scaffold_cols = ["time_end", "season"]
        elif "season" in groups:
            scaffold_cols = ["time_end"] + groups
        else:
            scaffold_cols = ["time_end", "season"] + groups

        scaffold = LPLModel.augment_scaffold(
            test_data.select(scaffold_cols).unique(),
            season_start_month,
            season_start_day,
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
            assert self.value_to_index is not None
            group_codes = iup.utils.value_to_index(
                scaffold.select(groups), self.value_to_index, self.num_group_levels
            )

            # Make a prediction-machine from the fit model
            predictions = np.array(
                predictive(
                    self.pred_key,
                    elapsed=scaffold["elapsed"].to_numpy(),
                    N_tot=scaffold["N_tot"].to_numpy(),
                    groups=group_codes,
                    num_group_factors=self.num_group_factors,
                    num_group_levels=self.num_group_levels,
                )["obs"]
            ).transpose()
        else:
            predictions = np.array(
                predictive(self.pred_key, elapsed=scaffold["elapsed"].to_numpy())["obs"]
            ).transpose()

        pred = pl.concat(
            [
                scaffold,
                pl.DataFrame(
                    predictions,
                    schema=[f"{i + 1}" for i in range(predictions.shape[1])],
                ),
            ],
            how="horizontal",
        )

        pred = (
            pred.unpivot(
                index=scaffold.columns,
                variable_name="sample_id",
                value_name="estimate",
            )
            .with_columns(
                sample_id=pl.col("sample_id"),
                estimate=pl.col("estimate") / pl.col("N_tot"),
            )
            .drop(["elapsed", "N_tot"])
        )

        return SampleForecast(pred)


def extract_group_combos(
    data: pl.DataFrame, groups: List[str,] | None
) -> pl.DataFrame | None:
    """Extract from uptake data all combinations of grouping factors.

    Args:
        data: Uptake data possibly containing grouping factors.
        groups: Names of the columns for the grouping factors.

    Returns:
        All combinations of grouping factors.

    Note:
        This is required by multiple models.
    """
    if groups is not None:
        return data.select(groups).unique()
    else:
        return None
