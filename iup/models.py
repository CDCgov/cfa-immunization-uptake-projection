import abc
import datetime as dt
from typing import List

import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
from typing_extensions import Self

import iup.utils
from iup import CumulativeUptakeData, IncidentUptakeData, SampleForecast, UptakeData


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
        groups: List[str] | None,
        rollouts: List[dt.date] | None,
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
        test_data: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Use a fit model to fill in forecasts in a scaffold of dates.
        """
        pass


class LinearIncidentUptakeModel(UptakeModel):
    """
    Subclass of UptakeModel for a linear model constructed as follows:
    Outcome: daily-average uptake for the interval preceding a report date
    Predictors:
    - number of days "elapsed" between rollout and the report date
    - daily-average uptake for the interval preceding the "previous" report date
    - interaction of "elapsed" and "previous"
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model.
        """
        self.rng_key = random.PRNGKey(seed)
        self.model = LinearIncidentUptakeModel.lium

    @staticmethod
    def lium(
        previous=None,
        elapsed=None,
        daily=None,
        a_mn=0.0,
        a_sd=1.0,
        bP_mn=0.0,
        bP_sd=1.0,
        bE_mn=0.0,
        bE_sd=1.0,
        bPE_mn=0.0,
        bPE_sd=1.0,
        sig_mn=1.0,
    ):
        """
        Declare the linear incident uptake model structure.

        Parameters
        previous: numpy array
            standardized values of previous daily average incident uptake, a predictor
        elapsed: numpy array
            standardized values of number of days since rollout, a predictor
        daily: numpy array
            standardized values of daily average incident uptake, the outcome
        other parameters: float
            means and standard deviations to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a linear incident uptake model.
        """
        a = numpyro.sample("a", dist.Normal(a_mn, a_sd))
        P, E, PE = 0.0, 0.0, 0.0
        if previous is not None:
            bP = numpyro.sample("bP", dist.Normal(bP_mn, bP_sd))
            P = bP * previous
        if elapsed is not None:
            bE = numpyro.sample("bE", dist.Normal(bE_mn, bE_sd))
            E = bE * elapsed
        if previous is not None and elapsed is not None:
            bPE = numpyro.sample("bEP", dist.Normal(bPE_mn, bPE_sd))
            PE = bPE * previous * elapsed
        sig = numpyro.sample("sig", dist.Exponential(sig_mn))
        mu = a + P + E + PE
        numpyro.sample("obs", dist.Normal(mu, sig), obs=daily)

    @staticmethod
    def extract_starting_conditions(
        data: IncidentUptakeData, groups: List[str,] | None
    ) -> pl.DataFrame:
        """
        From incident uptake training data, extract information from
        the last report date.

        Parameters:
        data: IncidentUptakeData
            training data for fitting a linear incident uptake model
        groups: List[str, ] | None
            column name(s) for grouping factors

        Returns:
            Data frame of information about last observed date.

        Details
        Information on the last report date is:
        - date
        - daily average incident uptake
        - days elapsed since rollout
        - cumulative uptake since rollout

        Even if no groups are specified, the data must at least be grouped by season.
        """
        if groups is None:
            groups = ["season"]

        start = (
            data.group_by(groups)
            .agg(
                [
                    pl.col("time_end").last().alias("last_date"),
                    pl.col("daily").last().alias("last_daily"),
                    pl.col("elapsed").last().alias("last_elapsed"),
                    pl.col("estimate").sum().alias("last_cumulative"),
                ]
            )
            .filter(pl.col("season") == pl.col("season").max())
        )

        return start

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
        groups: List[str] | None,
        rollouts: List[dt.date] | None,
    ) -> IncidentUptakeData:
        """
        Format preprocessed data for fitting a linear incident uptake model.

        Parameters:
        data: CumulativeUptakeData
            training data for fitting a linear incident uptake model
         season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season
        groups: List[str, ] | None
            column name(s) for grouping factors
        rollouts: List[dt.date] | None
            rollout dates for each season in the training data

        Returns:
            Incident uptake data ready for fitting a linear incident uptake model.

        Details
        The following steps are required to prepare preprocessed data
        for fitting a linear incident uptake model:
        - insert rollout dates
        - convert to incident uptake
        - add extra columns for
            - time elapsed since rollout
            - time interval between report dates
            - daily average incident uptake between report dates
            - daily average incident uptake between the two previous report dates
        """
        assert rollouts is not None, (
            "LinearIncidentUptakeModel requires rollout dates, but none provided"
        )

        data = data.insert_rollouts(
            rollouts, groups, season_start_month, season_start_day
        )

        incident_data = data.to_incident(groups)

        assert incident_data["time_end"].is_sorted(), (
            "Chronological sorting got broken during data augmentation!"
        )

        incident_data = LinearIncidentUptakeModel.augment_columns(incident_data, groups)

        return incident_data

    @staticmethod
    def augment_columns(
        data: IncidentUptakeData,
        groups: List[str] | None,
    ) -> IncidentUptakeData:
        """
        Add columns to data for fitting a linear incident uptake model.

        Parameters:
        data: IncidentUptakeData
            training data for fitting a linear incident uptake model
        groups: List[str, ] | None
            column name(s) for grouping factors

        Returns:
            Incident uptake data with extra columns required by the linear incident uptake model.

        Details
        Extra columns are added for
            - time elapsed since rollout
            - time interval between report dates
            - daily average incident uptake between report dates
            - daily average incident uptake between the two previous report dates
        """
        if groups is None:
            groups = ["season"]

        data = IncidentUptakeData(
            data.with_columns(
                elapsed=pl.col("time_end").pipe(iup.utils.date_to_elapsed).over(groups),
                interval=pl.col("time_end")
                .pipe(iup.utils.date_to_interval)
                .over(groups),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1).over(groups))
        )

        return data

    def fit(
        self,
        data: IncidentUptakeData,
        groups: List[str,] | None,
        params: dict,
        mcmc: dict,
    ) -> Self:
        """
        Fit a linear incident uptake model on training data.

        Parameters
        data: IncidentUptakeData
            training data on which to fit the model
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        params: dict
            parameter names and values to specify prior distributions
        mcmc: dict
            control parameters for mcmc fitting

        Returns
        LinearIncidentUptakeModel
            model object with projection starting conditions, standardization
            constants, and the model fit all stored as attributes

        Details
        Extra columns for fitting this model are added to the incident data,
        including daily-average uptake. This is modeled rather than total uptake
        to account for slight variations in interval lengths (e.g. 6 vs. 7 days).

        To enable projections later on, some starting conditions as well as
        standardization constants for the model's outcome and first-order predictors
        are recorded and stored as model attributes.

        If the training data spans multiple (combinations of) groups,
        complete pooling will be used to recognize the groups as distinct but
        to assume they behave identically except for initial conditions.

        Finally, the model is fit using numpyro.
        """
        self.group_combos = extract_group_combos(data, groups)

        self.standards = iup.utils.extract_standards(
            data, ("previous", "elapsed", "daily")
        )

        self.start = self.extract_starting_conditions(data, groups)

        data = IncidentUptakeData(
            data.trim_outlier_intervals(groups).with_columns(
                previous_std=pl.col("previous").pipe(iup.utils.standardize),
                elapsed_std=pl.col("elapsed").pipe(iup.utils.standardize),
                daily_std=pl.col("daily").pipe(iup.utils.standardize),
            )
        )

        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.rng_key,
            previous=data["previous_std"].to_numpy(),
            elapsed=data["elapsed_std"].to_numpy(),
            daily=data["daily_std"].to_numpy(),
            a_mn=params["a_mn"],
            a_sd=params["a_sd"],
            bP_mn=params["bP_mn"],
            bP_sd=params["bP_sd"],
            bE_mn=params["bE_mn"],
            bE_sd=params["bE_sd"],
            bPE_mn=params["bPE_mn"],
            bPE_sd=params["bPE_sd"],
            sig_mn=params["sig_mn"],
        )

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame,
        groups: List[str] | None,
        start: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a linear incident uptake model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        groups: List[str, ] | None
            column name(s) for grouping factors
        start: pl.DataFrame
            information about the last observed report date,
            which guides the first forecast

        Returns:
            Scaffold with extra columns required by the linear incident uptake model.

        Details
        Extra columns are added for
            - time elapsed since rollout
            - time interval between report dates
        Columns are removed for
            - estimated incident uptake
            - daily average incident uptake between report dates
            - daily average incident uptake between the two previous report dates
        """
        scaffold = LinearIncidentUptakeModel.augment_columns(
            IncidentUptakeData(scaffold), groups
        ).drop(["estimate", "daily", "previous"])

        if groups is not None:
            scaffold = scaffold.join(
                start.select(list(groups) + ["last_elapsed", "last_interval"]),
                on=groups,
            )
        else:
            scaffold = scaffold.with_columns(
                last_elapsed=start["last_elapsed"][0],
                last_interval=start["last_interval"][0],
            )

        scaffold = scaffold.with_columns(
            elapsed=pl.col("elapsed")
            + pl.col("last_elapsed")
            + pl.col("last_interval"),
            interval=pl.when(pl.col("interval").is_null())
            .then(pl.col("last_interval"))
            .otherwise(pl.col("interval")),
        ).drop(["last_elapsed", "last_interval"])

        return scaffold

    @classmethod
    def project_sequentially(
        cls, elapsed: tuple, start: float, standards: dict, model, mcmc, rng_key
    ) -> np.ndarray:
        """
        Perform sequential projections from a linear incident uptake model.

        Parameters
        elapsed: tuple
            days elapsed since rollout at each projection time point
        start: pl.DataFrame
            starting value for the first projection
        standards: dict
            means and standard deviations for the predictor and outcome variables
        model: Predictive
            fit model that predicts next daily-avg uptake from the current
        mcmc:
            MCMC samples from a fit linear incident uptake models
        rng_key:
            random seed

        Returns
        Projections over the desired time frame from a linear incident uptake model

        Details
        Because daily-average uptake (outcome) and previous daily-average
        uptake (predictor) each contain one observation that the other
        does not, the projection at each time point must be unstandardized
        according to the former and re-standardized according to the latter
        before it can be used to project the next time point. This is what
        necessitates the sequential nature of these projections.

        Projections are made separately by group, if grouping factors exist.
        This function handles one group at a time.
        """
        # Make a prediction machine using the fit model
        predictive = Predictive(model, mcmc.get_samples())

        # Array to hold the last known uptake and each sequential projection (cols)
        # for each MCMC sample in fit model (rows)
        proj = np.zeros((mcmc.get_samples()["a"].shape[0], len(elapsed) + 1))

        # First entry of each row is the last known uptake value
        proj[:, 0] = start

        # To make each sequential projection
        for i in range(proj.shape[1] - 1):
            # Predictors are standardized uptake on the previous projection date,
            # standardized days-elapsed on the current projection date, & interaction.
            prev = np.array(
                iup.utils.standardize(
                    proj[:, i],
                    standards["previous"]["mean"],
                    standards["previous"]["std"],
                )
            )
            elap = np.repeat(
                iup.utils.standardize(
                    elapsed[i],
                    standards["elapsed"]["mean"],
                    standards["elapsed"]["std"],
                ),
                proj.shape[0],
            )

            # Predict the uptake on the next date:
            # One prediction per MCMC sample (row) for each prev*elap combo (col)
            # The diagonal has the next date predicted with the same parameter draw
            # used to predict the input date.
            y = (predictive(rng_key, previous=prev, elapsed=elap)["obs"]).diagonal()

            # Unstandardize the projection onto its natural scale
            proj[:, i + 1] = iup.utils.unstandardize(
                y,
                standards["daily"]["mean"],
                standards["daily"]["std"],
            )
            # This projection becomes 'previous' in the next loop iteration.

        # Sequential projections become rows and sample trajectories cols.
        # Remove the first row, which is the last known uptake value.
        return proj.transpose()[1:, :]

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_data: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit linear incident uptake model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_data: pl.DataFrame | None
            test data, if evaluation is being done, to provide exact dates
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        LinearIncidentUptakeModel
            the model with incident and cumulative projections as attributes

        Details
        A scaffold is set up to house theprojections over the
        desired time window with the desired intervals.

        Starting conditions derived from the last observed date in the training data
        are used to project for the first date. From there, projections
        are generated sequentially, because the projection for each date
        depends on the previous date, thanks to the model structure.

        After projections are completed, they are converted from daily-average
        to total cumulative uptake, on each date.
        """
        scaffold = build_scaffold(
            start_date,
            end_date,
            interval,
            test_data,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        self.start = self.start.with_columns(
            last_interval=(
                scaffold["time_end"].min() - pl.col("last_date")
            ).dt.total_days()
        )

        scaffold = LinearIncidentUptakeModel.augment_scaffold(
            scaffold, groups, self.start
        )

        if groups is not None:
            combos = scaffold.partition_by(groups)
        else:
            combos = [scaffold]

        for g in range(len(combos)):
            if groups is not None:
                start = self.start.join(combos[g], on=groups, how="semi")["last_daily"][
                    0
                ]
            else:
                start = self.start["last_daily"][0]

            proj = self.project_sequentially(
                tuple(combos[g]["elapsed"]),
                start,
                self.standards,
                self.model,
                self.mcmc,
                self.rng_key,
            )

            proj = proj * combos[g]["interval"].to_numpy().reshape(-1, 1)

            if groups is not None:
                proj = (
                    np.cumsum(proj, 0)
                    + self.start.join(combos[g], on=groups, how="semi")[
                        "last_cumulative"
                    ][0]
                )
            else:
                proj = np.cumsum(proj, 0) + self.start["last_cumulative"][0]

            proj = pl.DataFrame(proj, schema=[f"{i + 1}" for i in range(proj.shape[1])])

            combos[g] = (
                pl.concat([combos[g], proj], how="horizontal")
                .unpivot(
                    index=combos[g].columns,
                    variable_name="sample_id",
                    value_name="estimate",
                )
                .with_columns(sample_id=pl.col("sample_id"))
            )

        cumulative_projection = pl.concat(combos)

        return SampleForecast(cumulative_projection)


class HillModel(UptakeModel):
    """
    Subclass of UptakeModel for a Hill function model constructed as follows:
    Outcome: cumulative uptake as of a report date
    Predictors:
    - number of days "elapsed" between the start of the season and the report date
    Possible Random Effects:
    - season
    """

    def __init__(self, seed: int):
        """
        Initialize with a seed and the model structure.

        Parameters
        seed: int
            The random seed for stochastic elements of the model.
        """
        self.rng_key = random.PRNGKey(seed)
        self.model = HillModel.hill

    @staticmethod
    def hill(
        elapsed,
        season=None,
        cum_uptake=None,
        n_low=1.0,
        n_high=5.0,
        A_low=0.0,
        A_high=1.0,
        A_sig=1.0,
        H_low=10.0,
        H_high=180.0,
        H_sig=1.0,
        sig_mn=1.0,
    ):
        """
        Fit a Hill model on training data.

        Parameters
        elapsed: np.array
            days elapsed since the season start for each data point
        season: np.array | None
            numeric code for the season each data point belongs to
        cum_uptake: np.array | None
            cumulative uptake measured at each data point
        other parameters: float
            means and standard deviations to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Hill model.
        """
        n = numpyro.sample("n", dist.Uniform(n_low, n_high))
        A = numpyro.sample("A", dist.Uniform(A_low, A_high))
        sig_A = numpyro.sample("sig_A", dist.Exponential(A_sig))
        H = numpyro.sample("H", dist.Uniform(H_low, H_high))
        sig_H = numpyro.sample("sig_H", dist.Exponential(H_sig))
        if season is not None:
            A_s = numpyro.sample(
                "A_s",
                dist.TruncatedNormal(A, sig_A, low=A_low, high=A_high),
                sample_shape=(len(np.unique(season)),),
            )
            H_s = numpyro.sample(
                "H_s",
                dist.TruncatedNormal(H, sig_H, low=H_low, high=H_high),
                sample_shape=(len(np.unique(season)),),
            )
            mu = A_s[season] * (elapsed**n) / (H_s[season] ** n + elapsed**n)
        else:
            mu = A * (elapsed**n) / (H**n + elapsed**n)
        sig = numpyro.sample("sig", dist.Exponential(sig_mn))
        numpyro.sample("obs", dist.Normal(mu, sig), obs=cum_uptake)

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
        groups: List[str] | None,
        rollouts: List[dt.date] | None,
    ) -> CumulativeUptakeData:
        data = CumulativeUptakeData(
            data.with_columns(
                elapsed=iup.utils.date_to_elapsed(
                    pl.col("time_end"),
                    season_start_month,
                    season_start_day,
                )
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
        Fit a hill model on training data.

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
        HillModel
            model object with grouping factor combinaions
            and the model fit all stored as attributes

        Details
        If season is provided as a grouping factor for the training data,
        a hierarchical model will be built with season-specific parameters
        for maximum uptake and half-maximal time, drawn from a shared distribution.
        Season is recoded numerically in this case, and the code for the last
        season (in which training data leaves off and forecasts begin)
        is recorded as a model attribute.

        If season is omitted as a grouping factor for the training data,
        all data are pooled to fit single parameters for maximum uptake
        and half-maximal time.

        The Hill exponent is always a single non-hierarchical parameter.

        Finally, the model is fit using numpyro.
        """
        self.group_combos = extract_group_combos(data, groups)

        if groups is None:
            groups = []

        if "season" in groups:
            season = data["season"].to_numpy()
            unique_seasons = np.unique(season)
            season_to_index = {s: i for i, s in enumerate(unique_seasons)}
            season = np.array([season_to_index[s] for s in season])
            self.season = season
        else:
            season = None

        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        self.mcmc.run(
            self.rng_key,
            cum_uptake=data["estimate"].to_numpy(),
            elapsed=data["elapsed"].to_numpy(),
            season=season,
            n_low=params["n_low"],
            n_high=params["n_high"],
            A_low=params["A_low"],
            A_high=params["A_high"],
            H_low=params["H_low"],
            H_high=params["H_high"],
            sig_mn=params["sig_mn"],
        )

        return self

    @staticmethod
    def augment_scaffold(
        scaffold: pl.DataFrame, season_start_month: int, season_start_day: int
    ) -> pl.DataFrame:
        """
        Add columns to a scaffold of dates for forecasting from a Hill model.

        Parameters:
        scaffold: pl.DataFrame
            scaffold of dates for forecasting
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns:
            Scaffold with extra columns required by the Hill model.

        Details
        An extra column is added for the time elapsed since the season start.
        That is all that's required to prepare the data for a Hill model.
        """
        scaffold = scaffold.with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
        ).drop("estimate")

        return scaffold

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_data: pl.DataFrame | None,
        groups: List[str,] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> pl.DataFrame:
        """
        Make projections from a fit hill model.

        Parameters
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        test_data: pl.DataFrame | None
            test data, if evaluation is being done, to provide exact dates
        groups: (str,) | None
            name(s) of the columns for the grouping factors
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        HillModel
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
            test_data,
            self.group_combos,
            season_start_month,
            season_start_day,
        )

        scaffold = HillModel.augment_scaffold(
            scaffold, season_start_month, season_start_day
        )

        predictive = Predictive(self.model, self.mcmc.get_samples())
        if groups is not None:
            predictions = np.array(
                predictive(
                    self.rng_key,
                    elapsed=scaffold["elapsed"].to_numpy(),
                    season=np.repeat(self.season.max(), scaffold.shape[0]),
                )["obs"]
            ).transpose()
        else:
            predictions = np.array(
                predictive(self.rng_key, elapsed=scaffold["elapsed"].to_numpy())["obs"]
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
                estimate=pl.col("estimate").cast(pl.Float64),
            )
            .drop("elapsed")
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
    group_cols: (str,) | None
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
    test_data: pl.DataFrame | None,
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
    test_data: pl.DataFrame | None
        test data, if evaluation is being done, to provide exact dates
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
    if test_data is not None:
        scaffold = (
            test_data.filter((pl.col("time_end").is_between(start_date, end_date)))
            .select(["time_end", "season"])
            .with_columns(estimate=pl.lit(0.0))
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
        scaffold = scaffold.join(group_combos, how="cross")

    return scaffold
