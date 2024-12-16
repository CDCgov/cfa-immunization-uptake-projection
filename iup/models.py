import abc
import datetime as dt
from typing import List

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from typing_extensions import Self

from iup import CumulativeUptakeData, IncidentUptakeData, UptakeData


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

    def __init__(self):
        """
        Initialize the model as a scikit-learn linear regression.
        """
        self.model = LinearRegression()

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
        start = data.group_by(group_cols).agg(
            [
                pl.col("date").last().alias("last_date"),
                pl.col("daily").last().alias("last_daily"),
                pl.col("elapsed").last().alias("last_elapsed"),
                (pl.col("estimate"))
                .filter(pl.col("season") == pl.col("season").max())
                .sum()
                .alias("last_cumulative"),
            ]
        )

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

    def fit(self, data: IncidentUptakeData, group_cols: List[str,] | None) -> Self:
        """
        Fit a linear incident uptake model on training data.

        Parameters
        data: IncidentUptakeData
            training data on which to fit the model
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors

        Returns
        LinearIncidentUptakeModel
            model object with projection starting conditions, standardization
            constants, predictor and outcome variables, and the model fit
            all stored as attributes

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

        Finally, the model is fit using the scikit-learn module.
        """
        # validate data
        data = IncidentUptakeData(data)

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

        self.x = (
            data.select(["previous_std", "elapsed_std"])
            .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
            .to_numpy()
        )

        self.y = data.select(["daily_std"]).to_numpy()

        self.model.fit(self.x, self.y)

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

    @classmethod
    def build_scaffold(
        cls,
        start: pl.DataFrame,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        group_cols: List[str,] | None,
    ) -> pl.DataFrame:
        """
        Build a scaffold data frame to hold projections of a linear incident uptake model.

        Parameters
        start: pl.DataFrame
            starting conditions for making projections
        start_date: dt.date
            the date on which projections should begin
        end_date: dt.date
            the date on which projections should end
        interval: str
            the time interval between projection dates,
            following timedelta convention (e.g. '7d' = seven days)
        group_cols: (str,) | None
            name(s) of the columns for the grouping factors

        Returns
        pl.DataFrame
            scaffold to hold model projections

        Details
        The desired time frame for projections is repeated over grouping factors,
        if any grouping factors exist.
        """
        scaffold = (
            pl.date_range(
                start=start_date,
                end=end_date,
                interval=interval,
                eager=True,
            )
            .alias("date")
            .to_frame()
            .with_columns(estimate=pl.lit(0.0))
        )

        if group_cols is not None:
            scaffold = scaffold.join(start.select(group_cols), how="cross")

        scaffold = cls.augment_implicit_columns(
            IncidentUptakeData(scaffold), group_cols
        )

        if group_cols is not None:
            scaffold = scaffold.join(
                start.select(list(group_cols) + ["last_elapsed", "last_interval"]),
                on=group_cols,
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
    def augment_implicit_columns(
        cls, df: IncidentUptakeData, group_cols: List[str,] | None
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
        return (
            IncidentUptakeData(df)
            .with_columns(
                season=pl.col("date").pipe(UptakeData.date_to_season),
                elapsed=pl.col("date").pipe(cls.date_to_elapsed).over(group_cols),
                interval=pl.col("date").pipe(cls.date_to_interval).over(group_cols),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1).over(group_cols))
        )

    @staticmethod
    def date_to_elapsed(date_col: pl.Expr) -> pl.Expr:
        """
        Extract a time elapsed column from a date column, as polars expressions.

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

    @classmethod
    def project_sequentially(
        cls, elapsed: tuple, start: float, standards: dict, model: LinearRegression
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
        model: LinearRegression
            model that predicts next daily-avg uptake from the current

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

        This does not yet incorporate uncertainty, but it should in the future.
        """
        # Vector to hold the last known uptake and each sequential projection
        proj = np.zeros(len(elapsed) + 1)

        # First entry is the last known uptake value
        proj[0] = start

        # To make each sequential projection
        for i in range(proj.shape[0] - 1):
            # Predictors are the standardized uptake on the previous projection date
            # and the standardized days-elapsed on the current projection date
            x = np.reshape(
                np.array(
                    [
                        cls.standardize(
                            proj[i],
                            standards["previous"]["mean"],
                            standards["previous"]["std"],
                        ),
                        cls.standardize(
                            elapsed[i],
                            standards["elapsed"]["mean"],
                            standards["elapsed"]["std"],
                        ),
                    ]
                ),
                (-1, 2),
            )
            # Include the interaction of the two 1st-order predictors
            x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
            # Predict the uptake on the projection date
            y = model.predict(x)
            # Unstandardize the projection onto its natural scale
            proj[i + 1] = cls.unstandardize(
                y[(0, 0)],
                standards["daily"]["mean"],
                standards["daily"]["std"],
            )
            # This projection becomes the previous projection
            # in the next loop iteration.

        return proj

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        group_cols: List[str,] | None,
    ) -> CumulativeUptakeData:
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
        self.start = self.start.with_columns(
            last_interval=(start_date - pl.col("last_date")).dt.total_days()
        )

        incident_projection = self.build_scaffold(
            self.start, start_date, end_date, interval, group_cols
        )

        if group_cols is not None:
            groups = incident_projection.partition_by(group_cols)
        else:
            groups = [incident_projection]

        for g in range(len(groups)):
            if group_cols is not None:
                start = self.start.join(groups[g], on=group_cols, how="semi")[
                    "last_daily"
                ][0]
            else:
                start = self.start["last_daily"][0]

            proj = self.project_sequentially(
                tuple(groups[g]["elapsed"]), start, self.standards, self.model
            )

            groups[g] = groups[g].with_columns(daily=pl.Series(np.delete(proj, 0)))

        incident_projection = pl.concat(groups).with_columns(
            estimate=pl.col("daily") * pl.col("interval")
        )

        incident_projection = IncidentUptakeData(incident_projection)

        if group_cols is not None:
            cumulative_projection = incident_projection.to_cumulative(
                group_cols, self.start.select(list(group_cols) + ["last_cumulative"])
            ).select(list(group_cols) + ["date", "estimate"])
        else:
            cumulative_projection = incident_projection.to_cumulative(
                group_cols, self.start.select(["last_cumulative"])
            ).select(["date", "estimate"])

        cumulative_projection = CumulativeUptakeData(cumulative_projection)

        return cumulative_projection

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
        rank = pl.col("date").rank().over(group_cols)
        shifted_standard_interval = (
            pl.col("date")
            .pipe(cls.date_to_interval)
            .pipe(cls.standardize)
            .shift(1)
            .over(group_cols)
        )

        return (
            # validate input
            IncidentUptakeData(df)
            # sort by date
            .sort("date")
            # keep only the correct rows
            .filter(
                (rank >= 4) | ((rank == 3) & (shifted_standard_interval <= threshold))
            )
        )

    @staticmethod
    def date_to_interval(date_col: pl.Expr) -> pl.Expr:
        """
        Extract a time interval column from a date column, as polars expressions.

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
