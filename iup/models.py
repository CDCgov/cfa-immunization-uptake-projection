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

from iup import CumulativeUptakeData, IncidentUptakeData, SampleForecast, UptakeData


class UptakeModel(abc.ABC):
    """
    Abstract class for different types of models.
    """

    @abc.abstractmethod
    def fit(self, data: IncidentUptakeData) -> Self:
        pass

    @abc.abstractmethod
    def predict(self, data: IncidentUptakeData, *args, **kwargs) -> IncidentUptakeData:
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
        data: IncidentUptakeData, group_cols: List[str,] | None
    ) -> pl.DataFrame:
        """
        Extract from incident uptake data the last observed values of several variables, by group.

        Parameters
        data: IncidentUptakeData
            incident uptake data containing final observations of interest
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors

        Returns
        pl.DataFrame
            the last observed values of several IncidentUptakeData variables

        Details
        Starting conditions include:
        - Last date on which uptake was observed
        - Daily average incident uptake on this date
        - Days elapsed since rollout on this date
        - Cumulative uptake since rollout on this date
        """
        if group_cols is not None:
            start = (
                data.group_by(group_cols)
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
        else:
            start = data.select(
                [
                    pl.col("time_end").last().alias("last_date"),
                    pl.col("daily").last().alias("last_daily"),
                    pl.col("elapsed").last().alias("last_elapsed"),
                    pl.col("estimate").sum().alias("last_cumulative"),
                ]
            ).filter(pl.col("season") == pl.col("season").max())

        return start

    @staticmethod
    def extract_standards(data: IncidentUptakeData, var_cols: tuple) -> dict:
        """
        Extract means and standard deviations from columns of incident uptake data.

        Parameters
        data: IncidentUptakeData
            incident uptake data with some columns to be standardized
        var_cols: (str,)
            column names of variables to be standardized

        Returns
        dict
            means and standard deviations for each variable column

        Details
        Keys are the variable names, and values are themselves
        dictionaries of mean and standard deviation.
        """
        standards = {
            var: {"mean": data[var].mean(), "std": data[var].std()} for var in var_cols
        }

        return standards

    @staticmethod
    def augment_data(
        data: CumulativeUptakeData,
        season_start_month: int,
        season_start_day: int,
        groups: List[str] | None,
        rollouts: List[dt.date] | None,
    ) -> IncidentUptakeData:
        assert rollouts is not None, (
            "LinearIncidentUptakeModel requires rollout dates, but none provided"
        )

        data = data.insert_rollouts(
            rollouts, groups, season_start_month, season_start_day
        )

        incident_data = data.to_incident(groups)

        return incident_data

    def fit(
        self,
        data: IncidentUptakeData,
        group_cols: List[str,] | None,
        params: dict,
        mcmc: dict,
    ) -> Self:
        """
        Fit a linear incident uptake model on training data.

        Parameters
        data: IncidentUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
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
        self.group_combos = extract_group_combos(data, group_cols)

        data = IncidentUptakeData(self.augment_implicit_columns(data, group_cols))

        self.start = self.extract_starting_conditions(data, group_cols)

        data = IncidentUptakeData(
            self.trim_outlier_intervals(data, group_cols).with_columns(
                previous_std=pl.col("previous").pipe(self.standardize),
                elapsed_std=pl.col("elapsed").pipe(self.standardize),
                daily_std=pl.col("daily").pipe(self.standardize),
            )
        )

        self.standards = self.extract_standards(data, ("previous", "elapsed", "daily"))

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
    def standardize(x, mn=None, sd=None):
        """
        Standardize: subtract mean and divide by standard deviation.

        Parameters
        x: pl.Expr | float64
            the numbers to standardize
        mn: float64
            the term to subtract, if not the mean of x
        sd: float64
            the term to divide by, if not the standard deviation of x

        Returns
        pl.Expr | float
            the standardized numbers

        Details
        If the standard deviation is 0, all standardized values are 0.0.
        """
        if type(x) is pl.Expr:
            if mn is not None:
                return (x - mn) / sd
            else:
                return (
                    pl.when(x.drop_nulls().n_unique() == 1)
                    .then(0.0)
                    .otherwise((x - x.mean()) / x.std())
                )
        else:
            if mn is not None:
                return (x - mn) / sd
            else:
                return (x - x.mean()) / x.std()

    @staticmethod
    def unstandardize(x, mn, sd):
        """
        Unstandardize: add standard deviation and multiply by mean.

        Parameters
        x: pl.Expr
            the numbers to unstandardize
        mn: float64
            the term to add
        sd: float64
            the term to multiply by

        Returns
        pl.Expr
            the unstandardized numbers
        """
        return x * sd + mn

    @staticmethod
    def augment_implicit_columns(
        df: IncidentUptakeData, group_cols: List[str,] | None
    ) -> pl.DataFrame:
        """
        Add explicit columns for information that is implicitly contained.

        Parameters
        data: IncidentUptakeData
            data containing dates and incident uptake estimates
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors

        Returns
        IncidentUptakeData
            data provided augmented with extra explicit columns.

        Details
        Extra columns are added to the incident uptake data:
        - disease season that each date belongs to
        - interval of time in days between each successive date
        - number of days elapsed between rollout and each date
        - daily-average uptake in the interval preceding each date
        - daily-average uptake in the interval preceding the previous date
        """
        assert df["time_end"].is_sorted(), (
            "Cannot perform 'date_to' operations if time_end is not chronologically sorted"
        )

        if group_cols is not None:
            data = (
                IncidentUptakeData(df)
                .with_columns(
                    elapsed=pl.col("time_end")
                    .pipe(LinearIncidentUptakeModel.date_to_elapsed)
                    .over(group_cols),
                    interval=pl.col("time_end")
                    .pipe(LinearIncidentUptakeModel.date_to_interval)
                    .over(group_cols),
                )
                .with_columns(daily=pl.col("estimate") / pl.col("interval"))
                .with_columns(previous=pl.col("daily").shift(1).over(group_cols))
            )
        else:
            data = (
                IncidentUptakeData(df)
                .with_columns(
                    elapsed=pl.col("time_end").pipe(
                        LinearIncidentUptakeModel.date_to_elapsed
                    ),
                    interval=pl.col("time_end").pipe(
                        LinearIncidentUptakeModel.date_to_interval
                    ),
                )
                .with_columns(daily=pl.col("estimate") / pl.col("interval"))
                .with_columns(previous=pl.col("daily").shift(1))
            )

        return data

    @staticmethod
    def date_to_elapsed(date_col: pl.Expr) -> pl.Expr:
        """
        Extract a time elapsed column from a date column, as polars expressions.
        This ought to be called .over(season)

        Parameters
        date_col: pl.Expr
            column of dates

        Returns
        pl.Expr
            column of the number of days elapsed since the first date

        Details
        Date column should be chronologically sorted in advance.
        Time difference is always in days.
        """

        return (date_col - date_col.first()).dt.total_days().cast(pl.Float64)

    @staticmethod
    def date_to_interval(date_col: pl.Expr) -> pl.Expr:
        """
        Extract a time interval column from a date column, as polars expressions.
        Should be called .over(season)

        Parameters
        date_col: pl.Expr
            column of dates

        Returns
        pl.Expr
            column of the number of days between each date and the previous

        Details
        Date column should be chronologically sorted in advance.
        Time difference is always in days.
        """
        return date_col.diff().dt.total_days().cast(pl.Float64)

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

        Returns
        IncidentUptakeProjection
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
                cls.standardize(
                    proj[:, i],
                    standards["previous"]["mean"],
                    standards["previous"]["std"],
                )
            )
            elap = np.repeat(
                cls.standardize(
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
            proj[:, i + 1] = cls.unstandardize(
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
        group_cols: List[str,] | None,
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

        Returns
        LinearIncidentUptakeModel
            the model with incident and cumulative projections as attributes

        Details
        A data frame is set up to house the incident projections over the
        desired time window with the desired intervals.

        The starting conditions derived from the last training data date
        are used to project for the first date. From there, projections
        are generated sequentially, because the projection for each date
        depends on the previous date, thanks to the model structure.

        After projections are completed, they are converted from daily-average
        to total incident uptake, as well as cumulative uptake, on each date.
        """
        # If there are test data, the actual start date for projections should
        # be the first date in the test data
        if test_data is not None:
            start_date = min(test_data["time_end"])

        self.start = self.start.with_columns(
            last_interval=(start_date - pl.col("last_date")).dt.total_days()
        )

        scaffold = build_scaffold(
            start_date, end_date, interval, test_data, self.group_combos
        ).drop("estimate")

        scaffold = LinearIncidentUptakeModel.augment_implicit_columns(
            IncidentUptakeData(scaffold, group_cols), group_cols
        )

        if group_cols is not None:
            scaffold = scaffold.join(
                self.start.select(list(group_cols) + ["last_elapsed", "last_interval"]),
                on=group_cols,
            )
        else:
            scaffold = scaffold.with_columns(
                last_elapsed=self.start["last_elapsed"][0],
                last_interval=self.start["last_interval"][0],
            )

        scaffold = scaffold.with_columns(
            elapsed=pl.col("elapsed")
            + pl.col("last_elapsed")
            + pl.col("last_interval"),
            interval=pl.when(pl.col("interval").is_null())
            .then(pl.col("last_interval"))
            .otherwise(pl.col("interval")),
        ).drop(["last_elapsed", "last_interval"])

        if group_cols is not None:
            groups = scaffold.partition_by(group_cols)
        else:
            groups = [scaffold]

        for g in range(len(groups)):
            if group_cols is not None:
                start = self.start.join(groups[g], on=group_cols, how="semi")[
                    "last_daily"
                ][0]
            else:
                start = self.start["last_daily"][0]

            proj = self.project_sequentially(
                tuple(groups[g]["elapsed"]),
                start,
                self.standards,
                self.model,
                self.mcmc,
                self.rng_key,
            )

            proj = proj * groups[g]["interval"].to_numpy().reshape(-1, 1)

            proj = (
                np.cumsum(proj, 0)
                + self.start.join(groups[g], on=group_cols, how="semi")[
                    "last_cumulative"
                ][0]
            )

            proj = pl.DataFrame(proj, schema=[f"{i + 1}" for i in range(proj.shape[1])])

            groups[g] = (
                pl.concat([groups[g], proj], how="horizontal")
                .unpivot(
                    index=groups[g].columns,
                    variable_name="sample_id",
                    value_name="estimate",
                )
                .with_columns(sample_id=pl.col("sample_id").cast(pl.Int64))
            )

        cumulative_projection = pl.concat(groups)

        return SampleForecast(cumulative_projection)

    @classmethod
    def trim_outlier_intervals(
        cls,
        df: IncidentUptakeData,
        group_cols: List[str,] | None,
        threshold: float = 1.0,
    ) -> pl.DataFrame:
        """
        Remove rows from incident uptake data with intervals that are too large.

        Parameters
          group_cols (tuple) | None: names of grouping factor columns
          threshold (float): maximum standardized interval between first two dates

        Returns
        pl.DataFrame:
            incident uptake data with the outlier rows removed

        Details
        The first row (index 0) is always rollout, so the second row (index 1)
        is the first actual report. Between these is often a long interval,
        compared to the fairly regular intervals among subsequent rows.

        If this interval is 1+ std dev bigger than the average interval, then
        the first report's incident uptake is likely an inflated outlier and
        should be excluded from the statistical model fitting.

        In this case, to fit a linear incident uptake model,
        the first three rows of the incident uptake data are dropped:
        - The first because it is rollout, where uptake is 0 (also an outlier)
        - The second because it is likely an outlier, as described above
        - The third because it's previous value is the second, an outlier

        Otherwise, only the first two rows of the incident uptake data are dropped:
        - The first because it is rollout, where uptake is 0 (also an outlier)
        - The second because it's previous value is 0, an outlier
        """
        assert df["time_end"].is_sorted(), (
            "Cannot perform 'date_to' operations if time_end is not chronologically sorted"
        )

        if group_cols is not None:
            rank = pl.col("time_end").rank().over(group_cols)
            shifted_standard_interval = (
                pl.col("time_end")
                .pipe(cls.date_to_interval)
                .pipe(cls.standardize)
                .shift(1)
                .over(group_cols)
            )
        else:
            rank = pl.col("time_end").rank()
            shifted_standard_interval = (
                pl.col("time_end")
                .pipe(cls.date_to_interval)
                .pipe(cls.standardize)
                .shift(1)
            )

        return (
            # validate input
            IncidentUptakeData(df)
            # sort by date
            .sort("time_end")
            # keep only the correct rows
            .filter(
                (rank >= 4) | ((rank == 3) & (shifted_standard_interval <= threshold))
            )
        )


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
        cum_uptake,
        elapsed,
        season=None,
        n_low=1.0,
        n_high=5.0,
        A_low=0.0,
        A_high=1.0,
        H_low=10.0,
        H_high=180.0,
        sig_mn=1.0,
    ):
        """
        Declare the Hill model structure.

        Parameters
        cum_uptake: numpy array
            cumulative uptake, between 0 and 1
        elapsed: numpy array
            number of days since the start of season
        season: numpy array
            season that each data point belongs to
        other parameters: float
            means and standard deviations to specify the prior distributions

        Returns
        Nothing

        Details
        Provides the model structure and priors for a Hill model.
        """
        n = numpyro.sample("n", dist.Uniform(n_low, n_high))
        A = numpyro.sample("A", dist.Uniform(A_low, A_high))
        H = numpyro.sample("H", dist.Uniform(H_low, H_high))
        if season is not None:
            mu = A[season] * (elapsed**n) / (H[season] ** n + elapsed**n)
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
                elapsed=HillModel.date_to_elapsed(
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
        group_cols: List[str,] | None,
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
        self.group_combos = extract_group_combos(data, group_cols)

        if group_cols is None:
            group_cols = []

        if "season" in group_cols:
            season = data["season"].to_numpy()
        else:
            season = None

        self.kernel = NUTS(self.model)
        self.mcmc = MCMC(
            self.kernel,
            num_warmup=mcmc["num_warmup"],
            num_samples=mcmc["num_samples"],
            num_chains=mcmc["num_chains"],
        )

        print(data["estimate"].to_numpy())
        print(data["elapsed"].to_numpy())
        print(season)

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
    def date_to_elapsed(
        date_col: pl.Expr, season_start_month: int, season_start_day: int
    ) -> pl.Expr:
        """
        Extract a time elapsed column from a date column, as polars expressions.

        Parameters
        date_col: pl.Expr
            column of dates
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        pl.Expr
            column of the number of days elapsed since the first date

        Details
        Time difference is always in days.
        """
        # for every date, figure out the season breakpoint in that year
        season_start = pl.date(date_col.dt.year(), season_start_month, season_start_day)

        # for dates before the season breakpoint in year, subtract a year
        year = date_col.dt.year()
        season_start_year = (
            pl.when(date_col < season_start).then(year - 1).otherwise(year)
        )

        # rewrite the season breakpoints to that immediately before each date
        season_start = pl.date(season_start_year, season_start_month, season_start_day)

        # return the number of days from season start to each date
        return (date_col - season_start).dt.total_days().cast(pl.Float64)

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        test_data: pl.DataFrame | None,
        group_cols: List[str,] | None,
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
        group_cols: (str,) | None
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
        """
        # If there are test data, the actual start date for projections should
        # be the first date in the test data
        if test_data is not None:
            start_date = min(test_data["time_end"])

        scaffold = build_scaffold(
            start_date, end_date, interval, test_data, self.group_combos
        ).drop("estimate")

        scaffold = scaffold.with_columns(
            elapsed=HillModel.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            ),
            season=UptakeData.date_to_season(
                pl.col("time_end"), season_start_month, season_start_day
            ),
        )

        # Left off here! Must produce Hill Model predictions, which won't require project_sequentially
        predictive = Predictive(self.model, self.mcmc.get_samples())
        predictions = predictive(
            self.rng_key,
            elapsed=scaffold["elapsed"].to_numpy(),
            season=scaffold["season"].to_numpy(),
        )["obs"]

        print(predictions)

        return predictions


def extract_group_combos(
    data: pl.DataFrame, group_cols: List[str,] | None
) -> pl.DataFrame | None:
    """
    Extract from cumulative uptake data all combinations of grouping factors.

    Parameters
    data: CumulativeUptakeData
        cumulative uptake data containing final observations of interest
    group_cols: (str,) | None
        name(s) of the columns for the grouping factors

    Returns
    pl.DataFrame
        all combinations of grouping factors
    """
    if group_cols is not None:
        return data.select(group_cols).unique()
    else:
        return None


def build_scaffold(
    start_date: dt.date,
    end_date: dt.date,
    interval: str,
    test_data: pl.DataFrame | None,
    group_combos: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Build a scaffold data frame to hold projections of a linear incident uptake model.

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

    Returns
    pl.DataFrame
        scaffold to hold model projections

    Details
    The desired time frame for projections is repeated over grouping factors,
    if any grouping factors exist.
    """
    # If there is test data such that evaluation will be performed,
    # use exactly the dates that are in the test data
    if test_data is not None:
        scaffold = (
            test_data.filter((pl.col("time_end").is_between(start_date, end_date)))
            .select("time_end")
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
            .with_columns(estimate=pl.lit(0.0))
        )

    if group_combos is not None:
        scaffold = scaffold.join(group_combos, how="cross")

    return scaffold
