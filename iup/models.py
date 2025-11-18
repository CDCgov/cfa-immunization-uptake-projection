import abc
import datetime as dt
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
        """
        Add columns to preprocessed uptake data to provide all
        input information that a specific model requires.
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
        """
        Fit a model on its properly augmented data.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def augment_scaffold(
        scaffold: pl.DataFrame,
        groups: List[str] | None,
        start: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting to provide all
        input information that a specific model requires.
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Use a fit model to fill in forecasts in a scaffold of dates.
        """
        pass

    # save for future models #
    mcmc = None


class PPRRModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Polynomial Plus Reciprocal Reduced model.
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.rng_key = random.key(seed)
        self.fit_key, self.pred_key = random.split(self.rng_key, 2)
        self.model = PPRRModel._poly_plus_recip_reduc

    @staticmethod
    def _poly_plus_recip_reduc(
        elapsed,
        N_vax=None,
        N_tot=None,
        groups=None,
        num_group_factors=0,
        num_group_levels=[0],
        b1_shape=1.0,
        b1_rate=1.0,
        b1_sig=1.0,
        n1_shape=1.0,
        n1_rate=1.0,
        b2_shape=1.0,
        b2_rate=1.0,
        b2_sig=1.0,
        n2_shape=1.0,
        n2_rate=1.0,
        a_shape1=1.0,
        a_shape2=1.0,
        a_sig=1.0,
        d_shape=350.0,
        d_rate=1.0,
    ):
        """
        Fit a mixed Polynomial Plus Reciprocal Reduced model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array | None
            number of people vaccinated at each data point
        N_tot: np.array | None
            number of people contacted at each data point
        groups: np.array | None
            numeric codes for groups: row = data point, col = grouping factor
        num_group_factors: Int
            number of grouping factors
        num_group_levels: List[Int,]
            number of unique levels of each grouping factor
        other parameters: float
            parameters to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Polynomial Plus Reciprocal Reduced model.
        """
        # Sample the overall average value for each parameter
        b1 = numpyro.sample("b1", dist.Gamma(b1_shape, b1_rate))
        n1 = numpyro.sample("n1", dist.Gamma(n1_shape, n1_rate))
        b2 = numpyro.sample("b2", dist.Gamma(b2_shape, b2_rate))
        n2 = numpyro.sample("n2", dist.Gamma(n2_shape, n2_rate))
        a = numpyro.sample("a", dist.Beta(a_shape1, a_shape2))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))
        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            b1_sigs = numpyro.sample(
                "b1_sigs", dist.Exponential(b1_sig), sample_shape=(num_group_factors,)
            )
            b2_sigs = numpyro.sample(
                "b2_sigs", dist.Exponential(b2_sig), sample_shape=(num_group_factors,)
            )
            a_sigs = numpyro.sample(
                "a_sigs", dist.Exponential(a_sig), sample_shape=(num_group_factors,)
            )
            b1_devs = numpyro.sample(
                "b1_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(b1_sigs, np.array(num_group_levels))
            b2_devs = numpyro.sample(
                "b2_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(b2_sigs, np.array(num_group_levels))
            a_devs = numpyro.sample(
                "a_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(a_sigs, np.array(num_group_levels))
            b1_tot = np.sum(b1_devs[groups], axis=1) + b1
            b2_tot = np.sum(b2_devs[groups], axis=1) + b2
            a_tot = np.sum(a_devs[groups], axis=1) + a
            # Calculate latent true uptake at each datum
            mu = a_tot + (b1_tot * elapsed**n1) - (a_tot / (1 + b2_tot * elapsed**n2))
        else:
            # Calculate latent true uptake at each datum
            mu = a + (b1 * elapsed**n1) - (a / (1 + b2 * elapsed**n2))
        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """
        Format preprocessed data for fitting a Polynomial Plus Reciprocal Reduced model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a Polynomial Plus Reciprocal Reduced model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Cumulative uptake data ready for fitting a Polynomial Plus Reciprocal Reduced model.

        Details
        The following steps are required to prepare preprocessed data:
        - Add an extra columns for time elapsed since start-of-season
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
        """
        Fit a mixed Polynomial Plus Reciprocal Reduced model on training data.

        Parameters
        data: CumulativeUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        PPRRModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
        If grouping factors are specified, a hierarchical model will be built with
        group-specific parameters for all free parameters.
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
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            b1_shape=params["b1_shape"],
            b1_rate=params["b1_rate"],
            b1_sig=params["b1_sig"],
            n1_shape=params["n1_shape"],
            n1_rate=params["n1_rate"],
            b2_shape=params["b2_shape"],
            b2_rate=params["b2_rate"],
            b2_sig=params["b2_sig"],
            n2_shape=params["n2_shape"],
            n2_rate=params["n2_rate"],
            a_shape1=params["a_shape1"],
            a_shape2=params["a_shape2"],
            a_sig=params["a_sig"],
            d_shape=params["d_shape"],
            d_rate=params["d_rate"],
        )

        print(self.mcmc.print_summary())

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Polynomial Plus Reciprocal Reduced model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Polynomial Splice Reciprocal Reduced model.

        Details
        An extra column is added for the time elapsed since the season start.
        Predictions are made as if 10,000 individuals are sampled.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=pl.lit(10000),
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit Polynomial Plus Reciprocal Reduced model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_dates: pl.DataFrame | None
            exact target dates to use, when test data exists
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        PPRRModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the projections over the
        desired time window with the desired intervals.

        Forecasts are the made for each date in this scaffold.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_dates,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = PPRModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
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


class PPRModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Polynomial Plus Reciprocal model.
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.rng_key = random.key(seed)
        self.fit_key, self.pred_key = random.split(self.rng_key, 2)
        self.model = PPRModel._poly_plus_recip

    @staticmethod
    def _poly_plus_recip(
        elapsed,
        N_vax=None,
        N_tot=None,
        groups=None,
        num_group_factors=0,
        num_group_levels=[0],
        b1_shape=1.0,
        b1_rate=1.0,
        b1_sig=1.0,
        n1_shape=1.0,
        n1_rate=1.0,
        n1_sig=1.0,
        b2_shape=1.0,
        b2_rate=1.0,
        b2_sig=1.0,
        n2_shape=1.0,
        n2_rate=1.0,
        n2_sig=1.0,
        a_shape1=1.0,
        a_shape2=1.0,
        a_sig=1.0,
        d_shape=350.0,
        d_rate=1.0,
    ):
        """
        Fit a mixed Polynomial Plus Reciprocal model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array | None
            number of people vaccinated at each data point
        N_tot: np.array | None
            number of people contacted at each data point
        groups: np.array | None
            numeric codes for groups: row = data point, col = grouping factor
        num_group_factors: Int
            number of grouping factors
        num_group_levels: List[Int,]
            number of unique levels of each grouping factor
        other parameters: float
            parameters to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Logistic Splice Linear model.
        """
        # Sample the overall average value for each parameter
        b1 = numpyro.sample("b1", dist.Gamma(b1_shape, b1_rate))
        n1 = numpyro.sample("n1", dist.Gamma(n1_shape, n1_rate))
        b2 = numpyro.sample("b2", dist.Gamma(b2_shape, b2_rate))
        n2 = numpyro.sample("n2", dist.Gamma(n2_shape, n2_rate))
        a = numpyro.sample("a", dist.Beta(a_shape1, a_shape2))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))
        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            b1_sigs = numpyro.sample(
                "b1_sigs", dist.Exponential(b1_sig), sample_shape=(num_group_factors,)
            )
            n1_sigs = numpyro.sample(
                "n1_sigs", dist.Exponential(n1_sig), sample_shape=(num_group_factors,)
            )
            b2_sigs = numpyro.sample(
                "b2_sigs", dist.Exponential(b2_sig), sample_shape=(num_group_factors,)
            )
            n2_sigs = numpyro.sample(
                "n2_sigs", dist.Exponential(n2_sig), sample_shape=(num_group_factors,)
            )
            a_sigs = numpyro.sample(
                "a_sigs", dist.Exponential(a_sig), sample_shape=(num_group_factors,)
            )
            b1_devs = numpyro.sample(
                "b1_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(b1_sigs, np.array(num_group_levels))
            n1_devs = numpyro.sample(
                "n1_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(n1_sigs, np.array(num_group_levels))
            b2_devs = numpyro.sample(
                "b2_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(b2_sigs, np.array(num_group_levels))
            n2_devs = numpyro.sample(
                "n2_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(n2_sigs, np.array(num_group_levels))
            a_devs = numpyro.sample(
                "a_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(a_sigs, np.array(num_group_levels))
            b1_tot = np.sum(b1_devs[groups], axis=1) + b1
            n1_tot = np.sum(n1_devs[groups], axis=1) + n1
            b2_tot = np.sum(b2_devs[groups], axis=1) + b2
            n2_tot = np.sum(n2_devs[groups], axis=1) + n2
            a_tot = np.sum(a_devs[groups], axis=1) + a
            # Calculate latent true uptake at each datum
            mu = (
                a_tot
                + (b1_tot * elapsed**n1_tot)
                - (a_tot / (1 + b2_tot * elapsed**n2_tot))
            )
        else:
            # Calculate latent true uptake at each datum
            mu = a + (b1 * elapsed**n1) - (a / (1 + b2 * elapsed**n2))
        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """
        Format preprocessed data for fitting a Polynomial Plus Reciprocal model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a Polynomial Plus Reciprocal model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Cumulative uptake data ready for fitting a Polynomial Plus Reciprocal model.

        Details
        The following steps are required to prepare preprocessed data:
        - Add an extra columns for time elapsed since start-of-season
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
        """
        Fit a mixed Polynomial Plus Reciprocal model on training data.

        Parameters
        data: CumulativeUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        PPRModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
        If grouping factors are specified, a hierarchical model will be built with
        group-specific parameters for all free parameters.
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
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            b1_shape=params["b1_shape"],
            b1_rate=params["b1_rate"],
            b1_sig=params["b1_sig"],
            n1_shape=params["n1_shape"],
            n1_rate=params["n1_rate"],
            n1_sig=params["n1_sig"],
            b2_shape=params["b2_shape"],
            b2_rate=params["b2_rate"],
            b2_sig=params["b2_sig"],
            n2_shape=params["n2_shape"],
            n2_rate=params["n2_rate"],
            n2_sig=params["n2_sig"],
            a_shape1=params["a_shape1"],
            a_shape2=params["a_shape2"],
            a_sig=params["a_sig"],
            d_shape=params["d_shape"],
            d_rate=params["d_rate"],
        )

        print(self.mcmc.print_summary())

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Polynomial Plus Reciprocal model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Polynomial Splice Reciprocal model.

        Details
        An extra column is added for the time elapsed since the season start.
        Predictions are made as if 10,000 individuals are sampled.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=pl.lit(10000),
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit Polynomial Plus Reciprocal model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_dates: pl.DataFrame | None
            exact target dates to use, when test data exists
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        PPRModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the projections over the
        desired time window with the desired intervals.

        Forecasts are the made for each date in this scaffold.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_dates,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = PPRModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
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


class LSLIModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Logistic Splice Linear Intercept model.
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.rng_key = random.key(seed)
        self.fit_key, self.pred_key = random.split(self.rng_key, 2)
        self.model = LSLIModel._logistic_splice_linear_intercept

    @staticmethod
    def _logistic_splice_linear_intercept(
        elapsed,
        N_vax=None,
        N_tot=None,
        groups=None,
        num_group_factors=0,
        num_group_levels=[0],
        a_shape1=100.0,
        a_shape2=180.0,
        a_sig=40.0,
        c_center=-5.0,
        c_spread=1.0,
        c_sig=5.0,
        h_shape1=100.0,
        h_shape2=225.0,
        n_shape=25.0,
        n_rate=1.0,
        n_sig=40.0,
        k_shape1=225.0,
        k_shape2=225.0,
        k_sig=40.0,
        d_shape=350.0,
        d_rate=1.0,
    ):
        """
        Fit a mixed Logistic Splice Linear model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array | None
            number of people vaccinated at each data point
        N_tot: np.array | None
            number of people contacted at each data point
        groups: np.array | None
            numeric codes for groups: row = data point, col = grouping factor
        num_group_factors: Int
            number of grouping factors
        num_group_levels: List[Int,]
            number of unique levels of each grouping factor
        other parameters: float
            parameters to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Logistic Splice Linear model.
        """
        # Sample the overall average value for each parameter
        a = numpyro.sample("a", dist.Beta(a_shape1, a_shape2))
        h = numpyro.sample("h", dist.Beta(h_shape1, h_shape2))
        n = numpyro.sample("n", dist.Gamma(n_shape, n_rate))
        c = numpyro.sample("c", dist.Normal(c_center, c_spread))
        k = numpyro.sample("k", dist.Beta(k_shape1, k_shape2))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))
        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            a_sigs = numpyro.sample(
                "a_sigs", dist.Exponential(a_sig), sample_shape=(num_group_factors,)
            )
            k_sigs = numpyro.sample(
                "k_sigs", dist.Exponential(k_sig), sample_shape=(num_group_factors,)
            )
            n_sigs = numpyro.sample(
                "n_sigs", dist.Exponential(n_sig), sample_shape=(num_group_factors,)
            )
            c_sigs = numpyro.sample(
                "c_sigs", dist.Exponential(c_sig), sample_shape=(num_group_factors,)
            )
            a_devs = numpyro.sample(
                "a_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(a_sigs, np.array(num_group_levels))
            k_devs = k_devs = numpyro.sample(
                "k_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(k_sigs, np.array(num_group_levels))
            n_devs = n_devs = numpyro.sample(
                "n_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(n_sigs, np.array(num_group_levels))
            c_devs = c_devs = numpyro.sample(
                "c_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(c_sigs, np.array(num_group_levels))
            a_tot = np.sum(a_devs[groups], axis=1) + a
            k_tot = np.sum(k_devs[groups], axis=1) + k
            n_tot = np.sum(n_devs[groups], axis=1) + n
            c_tot = jnp.exp(np.sum(c_devs[groups], axis=1) + c)
            # Calculate slope and intercept of the linear
            m_tot = ((a_tot - c_tot) * n_tot * jnp.exp(0 - n_tot * (k_tot - h))) / (
                (1 + jnp.exp(0 - n_tot * (k_tot - h))) ** 2
            )
            b_tot = (
                c_tot
                + (a_tot - c_tot) / (1 + jnp.exp(0 - n_tot * (k_tot - h)))
                - m_tot * k_tot
            )
            # Calculate latent true uptake at each datum
            mu = (
                c_tot + ((a_tot - c_tot) / (1 + jnp.exp(0 - n_tot * (elapsed - h))))
            ) * (elapsed <= k) + (m_tot * elapsed + b_tot) * (elapsed > k)
        else:
            # Calculate slope and intercept of the linear
            m = ((a - c) * n * jnp.exp(0 - n * (k - h))) / (
                (1 + jnp.exp(0 - n * (k - h))) ** 2
            )
            b = c + (a - c) / (1 + jnp.exp(0 - n * (k - h))) - m * k
            # Calculate latent true uptake at each datum
            mu = (c + ((a - c) / (1 + jnp.exp(0 - n * (elapsed - h))))) * (
                elapsed <= k
            ) + (m * elapsed + b) * (elapsed > k)
        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """
        Format preprocessed data for fitting a Logistic Splice Linear model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a Logistic Splice Linear model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Cumulative uptake data ready for fitting a Logistic Splice Linear model.

        Details
        The following steps are required to prepare preprocessed data
        for fitting a linear incident uptake model:
        - Add an extra columns for time elapsed since start-of-season
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
        """
        Fit a mixed Logistic Splice Linear model on training data.

        Parameters
        data: CumulativeUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        LSLModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
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
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            a_shape1=params["a_shape1"],
            a_shape2=params["a_shape2"],
            a_sig=params["a_sig"],
            c_center=params["c_center"],
            c_spread=params["c_spread"],
            c_sig=params["c_sig"],
            h_shape1=params["h_shape1"],
            h_shape2=params["h_shape2"],
            n_shape=params["n_shape"],
            n_rate=params["n_rate"],
            n_sig=params["n_sig"],
            k_shape1=params["k_shape1"],
            k_shape2=params["k_shape2"],
            k_sig=params["k_sig"],
            d_shape=params["d_shape"],
            d_rate=params["d_rate"],
        )

        print(self.mcmc.print_summary())

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Logistic Splice Linear model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Logistic Splice Linear model.

        Details
        An extra column is added for the time elapsed since the season start.
        Predictions are made as if 10,000 individuals are sampled.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=pl.lit(10000),
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit Logistic Splice Linear model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_dates: pl.DataFrame | None
            exact target dates to use, when test data exists
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        LSLModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the projections over the
        desired time window with the desired intervals.

        Forecasts are the made for each date in this scaffold.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_dates,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = LSLIModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
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


class LSLModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Logistic Splice Linear model.
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model, to be split for fitting vs. predicting.
        """
        self.rng_key = random.key(seed)
        self.fit_key, self.pred_key = random.split(self.rng_key, 2)
        self.model = LSLModel._logistic_splice_linear

    @staticmethod
    def _logistic_splice_linear(
        elapsed,
        N_vax=None,
        N_tot=None,
        groups=None,
        num_group_factors=0,
        num_group_levels=[0],
        a_shape1=100.0,
        a_shape2=180.0,
        a_sig=40.0,
        h_shape1=100.0,
        h_shape2=225.0,
        n_shape=25.0,
        n_rate=1.0,
        n_sig=40.0,
        k_shape1=225.0,
        k_shape2=225.0,
        k_sig=40.0,
        d_shape=350.0,
        d_rate=1.0,
    ):
        """
        Fit a mixed Logistic Splice Linear model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array | None
            number of people vaccinated at each data point
        N_tot: np.array | None
            number of people contacted at each data point
        groups: np.array | None
            numeric codes for groups: row = data point, col = grouping factor
        num_group_factors: Int
            number of grouping factors
        num_group_levels: List[Int,]
            number of unique levels of each grouping factor
        other parameters: float
            parameters to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Logistic Splice Linear model.
        """
        # Sample the overall average value for each parameter
        a = numpyro.sample("a", dist.Beta(a_shape1, a_shape2))
        h = numpyro.sample("h", dist.Beta(h_shape1, h_shape2))
        n = numpyro.sample("n", dist.Gamma(n_shape, n_rate))
        k = numpyro.sample("k", dist.Beta(k_shape1, k_shape2))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))
        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            a_sigs = numpyro.sample(
                "a_sigs", dist.Exponential(a_sig), sample_shape=(num_group_factors,)
            )
            k_sigs = numpyro.sample(
                "k_sigs", dist.Exponential(k_sig), sample_shape=(num_group_factors,)
            )
            n_sigs = numpyro.sample(
                "n_sigs", dist.Exponential(n_sig), sample_shape=(num_group_factors,)
            )
            a_devs = numpyro.sample(
                "a_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(a_sigs, np.array(num_group_levels))
            k_devs = k_devs = numpyro.sample(
                "k_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(k_sigs, np.array(num_group_levels))
            n_devs = n_devs = numpyro.sample(
                "n_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(n_sigs, np.array(num_group_levels))
            a_tot = np.sum(a_devs[groups], axis=1) + a
            k_tot = np.sum(k_devs[groups], axis=1) + k
            n_tot = np.sum(n_devs[groups], axis=1) + n
            # Calculate slope and intercept of the linear
            m_tot = (a_tot * n_tot * jnp.exp(0 - n_tot * (k_tot - h))) / (
                (1 + jnp.exp(0 - n_tot * (k_tot - h))) ** 2
            )
            b_tot = a_tot / (1 + jnp.exp(0 - n_tot * (k_tot - h))) - m_tot * k_tot
            # Calculate latent true uptake at each datum
            mu = (a_tot / (1 + jnp.exp(0 - n_tot * (elapsed - h)))) * (elapsed <= k) + (
                m_tot * elapsed + b_tot
            ) * (elapsed > k)
        else:
            # Calculate slope and intercept of the linear
            m = (a * n * jnp.exp(0 - n * (k - h))) / (
                (1 + jnp.exp(0 - n * (k - h))) ** 2
            )
            b = a / (1 + jnp.exp(0 - n * (k - h))) - m * k
            # Calculate latent true uptake at each datum
            mu = (a / (1 + jnp.exp(0 - n * (elapsed - h)))) * (elapsed <= k) + (
                m * elapsed + b
            ) * (elapsed > k)
        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """
        Format preprocessed data for fitting a Logistic Splice Linear model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a Logistic Splice Linear model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Cumulative uptake data ready for fitting a Logistic Splice Linear model.

        Details
        The following steps are required to prepare preprocessed data
        for fitting a linear incident uptake model:
        - Add an extra columns for time elapsed since start-of-season
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
        """
        Fit a mixed Logistic Splice Linear model on training data.

        Parameters
        data: CumulativeUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        LSLModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
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
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            a_shape1=params["a_shape1"],
            a_shape2=params["a_shape2"],
            a_sig=params["a_sig"],
            h_shape1=params["h_shape1"],
            h_shape2=params["h_shape2"],
            n_shape=params["n_shape"],
            n_rate=params["n_rate"],
            n_sig=params["n_sig"],
            k_shape1=params["k_shape1"],
            k_shape2=params["k_shape2"],
            k_sig=params["k_sig"],
            d_shape=params["d_shape"],
            d_rate=params["d_rate"],
        )

        print(self.mcmc.print_summary())

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Logistic Splice Linear model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Logistic Splice Linear model.

        Details
        An extra column is added for the time elapsed since the season start.
        Predictions are made as if 10,000 individuals are sampled.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=pl.lit(10000),
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit Logistic Splice Linear model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_dates: pl.DataFrame | None
            exact target dates to use, when test data exists
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        LSLModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the projections over the
        desired time window with the desired intervals.

        Forecasts are the made for each date in this scaffold.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_dates,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = LSLModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
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


class LPLModel(UptakeModel):
    """
    Subclass of UptakeModel for a mixed Logistic Plus Linear model.
    For details, see: <https://github.com/CDCgov/cfa-immunization-uptake-projection/blob/main/docs/model_details.md>
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model, to be split for fitting vs. predicting.
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
        A_shape1=100.0,
        A_shape2=180.0,
        A_sig=40.0,
        H_shape1=100.0,
        H_shape2=225.0,
        n_shape=25.0,
        n_rate=1.0,
        n_sig=40.0,
        M_shape=1.0,
        M_rate=10.0,
        M_sig=40.0,
        d_shape=350.0,
        d_rate=1.0,
    ):
        """
        Fit a mixed Logistic Plus Linear model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array | None
            number of people vaccinated at each data point
        N_tot: np.array | None
            number of people contacted at each data point
        groups: np.array | None
            numeric codes for groups: row = data point, col = grouping factor
        num_group_factors: Int
            number of grouping factors
        num_group_levels: List[Int,]
            number of unique levels of each grouping factor
        other parameters: float
            parameters to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Logistic Plus Linear model.
        """
        # Sample the overall average value for each parameter
        A = numpyro.sample("A", dist.Beta(A_shape1, A_shape2))
        H = numpyro.sample("H", dist.Beta(H_shape1, H_shape2))
        n = numpyro.sample("n", dist.Gamma(n_shape, n_rate))
        M = numpyro.sample("M", dist.Gamma(M_shape, M_rate))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))
        # If grouping factors are given, find the group-specific deviations for each datum
        if groups is not None:
            A_sigs = numpyro.sample(
                "A_sigs", dist.Exponential(A_sig), sample_shape=(num_group_factors,)
            )
            M_sigs = numpyro.sample(
                "M_sigs", dist.Exponential(M_sig), sample_shape=(num_group_factors,)
            )
            n_sigs = numpyro.sample(
                "n_sigs", dist.Exponential(n_sig), sample_shape=(num_group_factors,)
            )
            A_devs = numpyro.sample(
                "A_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(A_sigs, np.array(num_group_levels))
            M_devs = M_devs = numpyro.sample(
                "M_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(M_sigs, np.array(num_group_levels))
            n_devs = n_devs = numpyro.sample(
                "n_devs", dist.Normal(0, 1), sample_shape=(sum(num_group_levels),)
            ) * np.repeat(n_sigs, np.array(num_group_levels))
            A_tot = np.sum(A_devs[groups], axis=1) + A
            M_tot = np.sum(M_devs[groups], axis=1) + M
            n_tot = np.sum(n_devs[groups], axis=1) + n
            # Calculate latent true uptake at each datum
            mu = A_tot / (1 + jnp.exp(0 - n_tot * (elapsed - H))) + (M_tot * elapsed)
        else:
            # Calculate latent true uptake at each datum if no grouping factors
            mu = A / (1 + jnp.exp(0 - n * (elapsed - H))) + (M * elapsed)
        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
    ) -> CumulativeUptakeData:
        """
        Format preprocessed data for fitting a Logistic Plus Linear model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a Logistic Plus Linear model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Cumulative uptake data ready for fitting a Logistic Plus Linear model.

        Details
        The following steps are required to prepare preprocessed data
        for fitting a linear incident uptake model:
        - Add an extra columns for time elapsed since start-of-season
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
        """
        Fit a mixed Logistic Plus Linear model on training data.

        Parameters
        data: CumulativeUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        LPLModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
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
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.fit_key,
            elapsed=elapsed,
            N_vax=N_vax,
            N_tot=N_tot,
            groups=group_codes,
            num_group_factors=self.num_group_factors,
            num_group_levels=self.num_group_levels,
            A_shape1=params["A_shape1"],
            A_shape2=params["A_shape2"],
            A_sig=params["A_sig"],
            H_shape1=params["H_shape1"],
            H_shape2=params["H_shape2"],
            n_shape=params["n_shape"],
            n_rate=params["n_rate"],
            n_sig=params["n_sig"],
            M_shape=params["M_shape"],
            M_rate=params["M_rate"],
            M_sig=params["M_sig"],
            d_shape=params["d_shape"],
            d_rate=params["d_rate"],
        )

        print(self.mcmc.print_summary())

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Logistic Plus Linear model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Logistic Plus Linear model.

        Details
        An extra column is added for the time elapsed since the season start.
        Predictions are made as if 10,000 individuals are sampled.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
            N_tot=pl.lit(10000),
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_dates: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit Logistic Plus Linear model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_dates: pl.DataFrame | None
            exact target dates to use, when test data exists
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        LPLModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the projections over the
        desired time window with the desired intervals.

        Forecasts are the made for each date in this scaffold.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_dates,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = LPLModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())

        if groups is not None:
            # Make a numpy array of numeric codes for grouping factor levels
            # that matches the same codes used when fitting the model
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
    """
    Extract from uptake data all combinations of grouping factors.

    Parameters
    data: pl.DataFrame
        uptake data possibly containing grouping factors
    groups: (str,) | None
        name(s) of the columns for the grouping factors

    Returns
    pl.DataFrame
        all combinations of grouping factors

    Details
    This is required by multiple models.
    """
    if groups is not None:
        return data.select(groups).unique()
    else:
        return None


def build_scaffold(
    start_date: dt.date,
    end_date: dt.date,
    interval: str,
    test_dates: pl.DataFrame | None,
    group_combos: pl.DataFrame | None,
    season_start_month: int,
    season_start_day: int,
) -> pl.DataFrame:
    """
    Build a scaffold data frame to hold forecasts.

    Parameters
    start_date: dt.date
        the date on which projections should begin
    end_date: dt.date
        the date on which projections should end
    interval: str
        the time interval between projection dates,
        following timedelta convention (e.g. '7d' = seven days)
    test_dates pl.DataFrame | None
        exact target dates to use, when test data exists
    group_combos: pl.DataFrame | None
        all unique combinations of grouping factors in the data
    season_start_month: int
        first month of the overwinter disease season
    season_start_day: int
        first day of the first month of the overwinter disease season

    Returns
    pl.DataFrame
        scaffold to hold model forecasts.

    Details
    The desired time frame for projections is repeated over grouping factors,
    if any grouping factors exist. This is required by multiple models.
    """
    # If there is test data such that evaluation will be performed,
    # use exactly the dates that are in the test data
    if test_dates is not None:
        scaffold = (
            test_dates.filter((pl.col("time_end").is_between(start_date, end_date)))
            .with_columns(estimate=pl.lit(0.0))
            .unique()
        )
    # If there are no test data, use exactly the dates that were provided
    else:
        scaffold = (
            pl.date_range(
                start=start_date,
                end=end_date,
                interval=interval,
                eager=True,
            )
            .alias("time_end")
            .to_frame()
            .with_columns(
                estimate=pl.lit(0.0),
                season=iup.utils.date_to_season(
                    pl.col("time_end"), season_start_month, season_start_day
                ),
            )
        )

    if group_combos is not None:
        # Even if season is a grouping factor, predictions should not
        # be made for every season
        if "season" in group_combos.columns:
            group_combos = group_combos.drop("season").unique()

        # Only include grouping factors in the scaffold if season
        # wasn't the only grouping factor
        if group_combos.shape[1] > 0:
            scaffold = scaffold.join(group_combos, how="cross")

    return scaffold
