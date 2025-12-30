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

    @abc.abstractmethod
    def __init__(self, seed: int):
        pass

    # save for future models #
    mcmc = None


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

    @staticmethod
    def _logistic_plus_linear_groups(
        elapsed,
        N_vax,
        N_tot,
        data_level_matrix: np.ndarray,
        level_factor_matrix: np.ndarray,
        A_shape1,
        A_shape2,
        A_sig,
        H_shape1,
        H_shape2,
        n_shape,
        n_rate,
        M_shape,
        M_rate,
        M_sig,
        d_shape,
        d_rate,
    ):
        """
        Fit a mixed Logistic Plus Linear model on training data.

        Parameters
        elapsed: np.array
            fraction of a year elapsed since the start of season at each data point
        N_vax: np.array
            number of people vaccinated at each data point
        N_tot: np.array
            number of people contacted at each data point
        data_level_matrix:
            see iup.utils.get_design_matrices()
        level_factor_matrix:
            see iup.utils.get_design_matrices()
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

        _, n_levels = data_level_matrix.shape
        _, n_factors = level_factor_matrix.shape

        A_sigs = numpyro.sample(
            "A_sigs", dist.Exponential(A_sig), sample_shape=(n_factors,)
        )
        M_sigs = numpyro.sample(
            "M_sigs", dist.Exponential(M_sig), sample_shape=(n_factors,)
        )

        A_zs = numpyro.sample("A_zs", dist.Normal(0, 1), sample_shape=(n_levels,))
        M_zs = numpyro.sample("M_zs", dist.Normal(0, 1), sample_shape=(n_levels,))

        A_devs = numpyro.deterministic(
            "A_devs",
            jnp.matmul(level_factor_matrix, A_sigs) * A_zs,  # type: ignore
        )
        M_devs = numpyro.deterministic(
            "M_devs",
            jnp.matmul(level_factor_matrix, M_sigs) * M_zs,  # type: ignore
        )

        A_tot = A + jnp.matmul(data_level_matrix, A_devs)
        M_tot = M + jnp.matmul(data_level_matrix, M_devs)

        # Calculate latent true uptake at each datum
        mu = numpyro.deterministic(
            "mu", A_tot / (1 + jnp.exp(0 - n * (elapsed - H))) + (M_tot * elapsed)
        )

        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)  # type: ignore

    @staticmethod
    def _logistic_plus_linear_no_groups(
        elapsed,
        N_vax,
        N_tot,
        A_shape1,
        A_shape2,
        H_shape1,
        H_shape2,
        n_shape,
        n_rate,
        M_shape,
        M_rate,
        d_shape,
        d_rate,
    ):
        # Sample the overall average value for each parameter
        A = numpyro.sample("A", dist.Beta(A_shape1, A_shape2))
        H = numpyro.sample("H", dist.Beta(H_shape1, H_shape2))
        n = numpyro.sample("n", dist.Gamma(n_shape, n_rate))
        M = numpyro.sample("M", dist.Gamma(M_shape, M_rate))
        d = numpyro.sample("d", dist.Gamma(d_shape, d_rate))

        mu = numpyro.deterministic(
            "mu", A / (1 + jnp.exp(0 - n * (elapsed - H))) + (M * elapsed)
        )

        # Calculate the shape parameters for the beta-binomial likelihood
        S1 = mu * d
        S2 = (1 - mu) * d
        numpyro.sample("obs", dist.BetaBinomial(S1, S2, N_tot), obs=N_vax)  # type: ignore

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

        # prepare common run arguments for grouped and ungrouped models
        run_kwargs = {
            "elapsed": data["elapsed"].to_numpy(),
            "N_vax": data["N_vax"].to_numpy(),
            "N_tot": data["N_tot"].to_numpy(),
            "A_shape1": params["A_shape1"],
            "A_shape2": params["A_shape2"],
            "A_sig": params["A_sig"],
            "H_shape1": params["H_shape1"],
            "H_shape2": params["H_shape2"],
            "n_shape": params["n_shape"],
            "n_rate": params["n_rate"],
            "M_shape": params["M_shape"],
            "M_rate": params["M_rate"],
            "M_sig": params["M_sig"],
            "d_shape": params["d_shape"],
            "d_rate": params["d_rate"],
        }

        # Tranform the levels of the grouping factors into numeric codes
        if groups is not None:
            self.level_to_index = iup.utils.map_level_to_index(data.select(groups))
            data_level_matrix, level_factor_matrix = iup.utils.get_design_matrices(
                data.select(groups), self.level_to_index
            )

            run_kwargs |= {
                "data_level_matrix": data_level_matrix,
                "level_factor_matrix": level_factor_matrix,
                "A_sig": params["A_sig"],
                "M_sig": params["M_sig"],
            }

            model = self._logistic_plus_linear_groups

            self.kernel = NUTS(
                self._logistic_plus_linear_groups, init_strategy=init_to_sample
            )
        else:
            model = self._logistic_plus_linear_no_groups

        self.kernel = NUTS(model, init_strategy=init_to_sample)

        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(self.fit_key, **run_kwargs)

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
            assert self.level_to_index is not None
            data_level_matrix, level_factor_matrix = iup.utils.get_design_matrices(
                scaffold.select(groups), self.level_to_index
            )

            # Make a prediction-machine from the fit model
            predictions = np.array(
                predictive(
                    self.pred_key,
                    elapsed=scaffold["elapsed"].to_numpy(),
                    N_tot=scaffold["N_tot"].to_numpy(),
                    data_level_matrix=data_level_matrix,
                    level_factor_matrix=level_factor_matrix,
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
