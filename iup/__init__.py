import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import abc
from typing_extensions import Self
from typing import List
import re
from polars import testing


class ValidatedUptake(pl.DataFrame):
    """
    Abstract class for observed data and forecast data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        """
        Validate that an ValidatedUptake object has the two key columns:
        date and uptake estimate (% of population). There may be others.
        """

        self.assert_columns_found(["date", "estimate"])
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["estimate"], pl.Float64)

    def assert_columns_found(self, columns: List[str]):
        """
        Verify that expected columns are found.

        Parameters
        columns: List[str]
            names of expected columns
        """
        for col in columns:
            assert col in self.columns, f"Column {col} is expected but not found."

    def assert_columns_type(self, columns: List[str], dtype):
        """
        Verify that columns have the expected type.

        Parameters
        columns: List[str]
            names of columns for which to check type
        dtype:
            data type for each listed column
        """
        for col in columns:
            assert (
                self[col].dtype == dtype
            ), f"Column {col} should be {dtype} but is {self[col].dtype}"

    def assert_type_included(self, datatype):
        """
        Verify at least one column has the expected type.

        Parameters
        columns: List[str]
            names of columns for which to check type
        datatype:
            data type for each listed column
        """
        assert any(
            dtype == datatype for dtype in self.schema.values()
        ), f"No column is {datatype} type"

    def with_columns(self, *args, **kwargs):
        """
        Copy of polars with_columns that returns the same subclass as it's given.
        """
        orig_class = type(self)
        result = super().with_columns(*args, **kwargs)
        return orig_class(result)

    def assert_numerical_columns_name(self, column_name):
        """
        Verify that all the numerical columns have a pattern with a common name

        Parameters
        column_name:
            The common column name
        """
        estimate = self.select(pl.all().exclude([pl.Date, pl.String]))
        pattern = rf"^{column_name}\d*$"
        assert all(
            [bool(re.match(pattern, col)) for col in estimate.columns]
        ), f"Not all columns are Column name {column_name}"

    def assert_numerical_columns_type(self, dtype):
        """
        Verify that all the numerical columns are the same data type

        Parameters
        dtype:
            Required data type, should be numerical
        """
        estimate = self.select(pl.all().exclude([pl.Date, pl.String]))

        for col in estimate.columns:
            assert (
                estimate[col].dtype == dtype
            ), f"Column {col} should be {dtype} but is {estimate[col].dtype}"

    @staticmethod
    def assert_column_name_all(estimate, column_name):
        """
        Verify that all columns have a pattern with a common name

        Parameters
        estimate:
            The data without 'date' type
        column_name:
            The common column name
        """
        pattern = rf"^{column_name}\d+$"
        assert all(
            [bool(re.match(pattern, col)) for col in estimate.columns]
        ), f"Not all columns are Column name {column_name}"

    @staticmethod
    def date_to_season(date_col: pl.Expr) -> pl.Expr:
        """
        Extract season column from a date column, as polars expressions.

        Parameters
        date_col: pl.Expr
            column of dates

        Returns
        pl.Expr
            column of the season for each date

        Details
        Assume overwinter seasons, e.g. 2023-10-07 and 2024-04-18 are both in "2023/24"
        """
        year1 = (
            date_col.dt.year() + pl.when(date_col.dt.month() < 7).then(-1).otherwise(0)
        ).cast(pl.Utf8)
        year2 = (
            date_col.dt.year() + pl.when(date_col.dt.month() < 7).then(0).otherwise(1)
        ).cast(pl.Utf8)
        season = pl.concat_str([year1, year2], separator="/")

        return season

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
        interval = date_col.diff().dt.total_days().cast(pl.Float64)

        return interval

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
        elapsed = (date_col - date_col.first()).dt.total_days().cast(pl.Float64)

        return elapsed

    @staticmethod
    def split_train_test(uptake_data_list, start_date: dt.date, side: str):
        """
        Concatenate ValidatedUptake objects and split into training and test data.

        Parameters
        uptake_data_list: List[ValidatedUptake]
            cumulative or incident uptake data objects, often from different seasons
        start_date: dt.date
            the first date for which projections should be made
        side: str
            whether the "train" or "test" portion of the data is desired

        Returns
        ValidatedUptake
            cumulative or uptake data object of the training or test portion

        Details
        Training data are before the start date; test data are on or after.
        """
        orig_class = type(uptake_data_list[0])
        if side == "train":
            out = (
                pl.concat(uptake_data_list)
                .sort("date")
                .filter(pl.col("date") < start_date)
            )
        if side == "test":
            out = (
                pl.concat(uptake_data_list)
                .sort("date")
                .filter(pl.col("date") >= start_date)
            )

        out = orig_class(out)

        return out


class IncidentUptakeData(ValidatedUptake):
    """
    Subclass of ValidatedUptake for incident uptake.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trim_outlier_intervals(
        self, group_cols: tuple[str,] | None, threshold: float = 1.0
    ):
        """
        Remove rows from incident uptake data with intervals that are too large.

        Parameters
          group_cols (tuple) | None: names of grouping factor columns
          threshold (float): maximum standardized interval between first two dates

        Returns
        IncidentValidatedUptake
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
            .pipe(self.date_to_interval)
            .pipe(standardize)
            .shift(1)
            .over(group_cols)
        )

        df = self.sort("date")

        df = df.filter(
            (rank >= 4) | ((rank == 3) & (shifted_standard_interval <= threshold))
        )

        return IncidentUptakeData(df)

    def augment_implicit_columns(self, group_cols: tuple[str,] | None) -> Self:
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
        self = (
            self.with_columns(
                season=pl.col("date").pipe(super().date_to_season),
                elapsed=pl.col("date").pipe(super().date_to_elapsed).over(group_cols),
                interval=pl.col("date").pipe(super().date_to_interval).over(group_cols),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1).over(group_cols))
        )

        return self

    def to_cumulative(self, group_cols: tuple[str,] | None, last_cumulative=None):
        """
        Convert incident to cumulative uptake data.

        Parameters
        group_cols: (str,) | None
            name(s) of the columns of grouping factors
        last_cumulative: pl.DataFrame
            additional cumulative uptake absent from the incident data, for each group

        Returns
        CumulativeUptakeData
            cumulative uptake on each date in the input incident uptake data

        Details
        When building models, the first 2-3 rows of incident uptake may be removed,
        such that the sum of incident does not reflect the entire cumulative uptake.
        To fix this, cumulative uptake from the removed rows may be supplied separately.
        """
        out = self.with_columns(estimate=pl.col("estimate").cum_sum().over(group_cols))

        if last_cumulative is not None:
            if group_cols is not None:
                out = out.join(last_cumulative, on=group_cols)
            else:
                out = out.with_columns(
                    last_cumulative=last_cumulative["last_cumulative"][0]
                )

            out = out.with_columns(
                estimate=pl.col("estimate") + pl.col("last_cumulative")
            ).drop("last_cumulative")

        out = CumulativeUptakeData(out)

        return out


class CumulativeUptakeData(ValidatedUptake):
    """
    Subclass of ValidatedUptake for cumulative uptake.
    """

    def to_incident(self, group_cols: tuple[str,] | None) -> IncidentUptakeData:
        """
        Convert cumulative to incident uptake data.

        Parameters
        group_cols: (str,) | None
            name(s) of the columns of grouping factors

        Returns
        IncidentUptakeData
            incident uptake on each date in the input cumulative uptake data

        Details
        Because the first date for each group is rollout, incident uptake is 0.
        """
        out = self.with_columns(
            estimate=pl.col("estimate").diff().over(group_cols).fill_null(0)
        )

        out = IncidentUptakeData(out)

        return out


def parse_nis(
    path: str,
    estimate_col: str,
    date_col: str,
    group_cols: dict,
    date_format: str,
    rollout: dt.date,
    filters=None,
) -> CumulativeUptakeData:
    """
    Load and parse NIS cumulative uptake data from a source file or address.

    Parameters
    path: str
        file path or url address for NIS data to import
    estimate_col: str
        name of the NIS column for the uptake estimate (population %)
    date_col: str
        name of the NIS column for the date
    group_cols: dict
        dictionary of the NIS columns for the grouping factors
        keys are the NIS column names and values are the desired column names
    date_format: str
        format of the dates in the NIS date column
    rollout: dt.date
        date of rollout
    filters: dict | None
        filters to remove unnecessary rows from the NIS data
        keys are NIS column names and values are entries for rows to keep

    Returns
        CumulativeUptakeData
        NIS data parsed to just the cumulative uptake of interest

    Details
    Parsing includes the following steps:
    - import NIS data from source
    - apply filters if any are supplied
    - isolate the estimate, date, and any grouping factor columns
    - insert an initial entry of 0 uptake on the rollout date
    """
    frame = fetch_nis(path)

    if filters is not None:
        frame = apply_filters(frame, filters)

    frame = select_columns(frame, estimate_col, date_col, group_cols, date_format)

    frame = insert_rollout(frame, rollout, group_cols)

    frame = CumulativeUptakeData(frame)

    return frame


def fetch_nis(path: str) -> pl.DataFrame:
    """
    Get NIS data from its file path or address.

    Parameters
    path: str
        file path or url address for NIS data to import

    Returns
    pl.DataFrame
        NIS cumulative uptake data exactly as in its source
    """
    return pl.read_csv(path)


def apply_filters(frame: pl.DataFrame, filters: dict | None) -> pl.DataFrame:
    """
    Apply filters to NIS data to remove unnecessary rows.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    filters: dict | None
        filters to remove unnecessary rows from the NIS data
        keys are NIS column names and values are entries for rows to keep

    Returns
    pl.DataFrame
        NIS cumulative uptake data with unnecessary rows removed

    Details
        If multiple entries in a column indicate that a row should be kept,
        the dictionary value for that column may be a list of these entries.
    """
    if filters is not None:
        filter_expr = pl.lit(True)
        for k, v in filters.items():
            filter_expr &= pl.col(k).is_in(pl.lit(v))
        frame = frame.filter(filter_expr)

    return frame


def select_columns(
    frame: pl.DataFrame,
    estimate_col: str,
    date_col: str,
    group_cols: dict | None,
    date_format: str,
) -> pl.DataFrame:
    """
    Select the date, uptake, and any grouping columns of NIS uptake data.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    estimate_col: str
        name of the NIS column for the uptake estimate (population %)
    date_col: str
        name of the NIS column for the date
    group_cols: dict | None
        dictionary of the NIS columns for the grouping factors
        keys are the NIS column names and values are the desired column names
    date_format: str
        format of the dates in the NIS date column

    Returns
        NIS cumulative uptake data with only the necessary columns

    Details
        Only the estimate, date, and grouping factor columns are kept.
    """

    if group_cols is not None:
        frame = frame.select(
            [estimate_col] + [date_col] + [pl.col(k) for k in group_cols.keys()]
        ).rename(group_cols)
    else:
        frame = frame.select([estimate_col] + [date_col])

    frame = (
        frame.with_columns(
            estimate=pl.col(estimate_col).cast(pl.Float64, strict=False),
            date=pl.col(date_col).str.to_date(date_format),
        )
        .drop_nulls(subset=["estimate"])
        .sort("date")
    )

    return frame


def insert_rollout(
    frame: pl.DataFrame, rollout: dt.date, group_cols: dict | None
) -> pl.DataFrame:
    """
    Insert into NIS uptake data rows with 0 uptake on the rollout date.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    rollout: dt.date
        rollout date
    group_cols: dict | None
        dictionary of the NIS columns for the grouping factors
        keys are the NIS column names and values are the desired column names

    Returns
        NIS cumulative data with rollout rows included

    Details
    A separate rollout row is added for every grouping factor combination.
    """
    if group_cols is not None:
        rollout_rows = (
            frame.select(pl.col(v) for v in group_cols.values())
            .unique()
            .with_columns(date=rollout, estimate=0.0)
        )
    else:
        rollout_rows = pl.DataFrame({"date": rollout, "estimate": 0.0})

    frame = frame.vstack(rollout_rows.select(frame.columns)).sort("date")

    return frame


def extract_group_names(
    group_cols=[
        dict,
    ],
) -> tuple[str,] | None:
    """
    Insure that the column names for grouping factors match across data sets.

    Parameters
    group_cols: [dict,]
        List of dictionaries of grouping factor column names, where
        keys are the NIS column names and values are the desired column names

    Returns
        (str,)
        The desired column names

    Details
    Before returning a single tuple of the desired column names,
    check that they are identical for every entry in the dictionary,
    where each entry represents one data set.
    """

    if None in group_cols:
        group_names = None
    else:
        assert all([len(g) == len(group_cols[0]) for g in group_cols])
        assert all([set(g.values()) == set(group_cols[0].values()) for g in group_cols])
        group_names = tuple([v for v in group_cols[0].values()])

    return group_names


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
        data: IncidentUptakeData, group_cols: tuple[str,] | None
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

    def fit(self, data: IncidentUptakeData, group_cols: tuple[str,] | None) -> Self:
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
        data = data.augment_implicit_columns(group_cols)

        self.start = LinearIncidentUptakeModel.extract_starting_conditions(
            data, group_cols
        )

        data = data.trim_outlier_intervals(group_cols).with_columns(
            previous_std=pl.col("previous").pipe(standardize),
            elapsed_std=pl.col("elapsed").pipe(standardize),
            daily_std=pl.col("daily").pipe(standardize),
        )

        self.standards = LinearIncidentUptakeModel.extract_standards(
            data, ("previous", "elapsed", "daily")
        )

        self.x = (
            data.select(["previous_std", "elapsed_std"])
            .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
            .to_numpy()
        )

        self.y = data.select(["daily_std"]).to_numpy()

        self.model.fit(self.x, self.y)

        return self

    @staticmethod
    def build_scaffold(
        start: pl.DataFrame,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
        group_cols: tuple[str,] | None,
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

        scaffold = IncidentUptakeData(scaffold).augment_implicit_columns(group_cols)

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

    @staticmethod
    def project_sequentially(
        elapsed: tuple, start: float, standards: dict, model: LinearRegression
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
        according to the former and restandardized according to the latter
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
                        standardize(
                            proj[i],
                            standards["previous"]["mean"],
                            standards["previous"]["std"],
                        ),
                        standardize(
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
            proj[i + 1] = unstandardize(
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
        group_cols: tuple[str,] | None,
    ) -> Self:
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
        A data frame is set up to house the incident projectins over the
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

        self.incident_projection = LinearIncidentUptakeModel.build_scaffold(
            self.start, start_date, end_date, interval, group_cols
        )

        if group_cols is not None:
            groups = self.incident_projection.partition_by(group_cols)
        else:
            groups = [self.incident_projection]

        for g in range(len(groups)):
            if group_cols is not None:
                start = self.start.join(groups[g], on=group_cols, how="semi")[
                    "last_daily"
                ][0]
            else:
                start = self.start["last_daily"][0]

            proj = LinearIncidentUptakeModel.project_sequentially(
                tuple(groups[g]["elapsed"]), start, self.standards, self.model
            )

            groups[g] = groups[g].with_columns(daily=pl.Series(np.delete(proj, 0)))

        self.incident_projection = pl.concat(groups).with_columns(
            estimate=pl.col("daily") * pl.col("interval")
        )

        self.incident_projection = IncidentUptakeData(self.incident_projection)

        if group_cols is not None:
            self.cumulative_projection = self.incident_projection.to_cumulative(
                group_cols, self.start.select(list(group_cols) + ["last_cumulative"])
            ).select(list(group_cols) + ["date", "estimate"])
        else:
            self.cumulative_projection = self.incident_projection.to_cumulative(
                group_cols, self.start.select(["last_cumulative"])
            ).select(["date", "estimate"])

        self.cumulative_projection = CumulativeUptakeData(self.cumulative_projection)

        return self


#### prediction output ####
class QuantileForecast(ValidatedUptake):
    """
    Class for forecast with quantiles.
    Save for future.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Must be named as "quantileXX" except the date column
    def validate(self):
        # has at least 1 column of date and 1 column of estimate
        self.assert_type_included(pl.Date)
        self.assert_type_included(pl.Float64)

        # except date, must have the common column names
        estimate = self.select(pl.all().exclude(pl.Date))
        super().assert_column_name_all(estimate, "quantile")


class PointForecast(QuantileForecast):
    """
    Class for forecast with point estimate
    A subclass when quantile is 50%
    For now, enforce the "quantile50" to be "estimate"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self):
        super().assert_columns_found(["estimate"])
        super().assert_columns_type(["estimate"], pl.Float64)


class SampleForecast(ValidatedUptake):
    """
    Class for forecast with posterior distribution.
    Save for future.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self):
        # has at least 1 column of date and 1 column of estimate
        self.assert_type_included(pl.Date)
        self.assert_type_included(pl.Float64)

        # except date, must have the common column names
        estimate = self.select(pl.all().exclude(pl.Date))
        super().assert_column_name_all(estimate, "sample_id")


###### evaluation metrics #####
def check_date_match(data: IncidentUptakeData, pred: PointForecast):
    """
    Check the dates between data and pred.
    Dates must be 1-on-1 equal and no duplicate.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    Error if conditions fail to meet.

    """
    # sort data and pred by date #
    data = data.sort("date")
    pred = pred.sort("date")

    # 1. Dates must be 1-on-1 equal
    testing.assert_series_equal(data["date"], pred["date"])

    # 2. There should not be any duplicated date in either data or prediction.
    assert (
        len(data["date"]) == data["date"].n_unique()
    ), "Duplicated dates are found in data and prediction."


def calculate_mspe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate MSPE between observed data and forecast.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and MSPE

    """
    joined = data.join(pred, on="date", how="inner", validate="1:1")

    start = (
        joined.filter(
            pl.col("date") == pl.col("date").min(),
        )
        .rename({"date": "forecast_start"})
        .select("forecast_start")
    )

    end = (
        joined.rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(spe=(pl.col("data") - pl.col("pred")) ** 2)
        .with_columns(
            mspe=pl.col("spe").mean(),
        )
        .filter(pl.col("date") == pl.col("date").max())
        .rename({"date": "forecast_end"})
        .select("forecast_end", "mspe")
    )

    return pl.concat([start, end], how="horizontal")


def get_mspe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Wrapper function of check_date_match and calculate_mspe
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and MSPE

    """
    check_date_match(data, pred)
    mspe = calculate_mspe(data, pred)

    return mspe


def calculate_mean_bias(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate Mean bias from joined data.
    Note the bias here is not the classical bias calculated from the posterior distribution.

    The bias here is defined as: at time t,
    bias = -1 if pred_t < data_t; bias = 0 if pred_t == data_t; bias = 1 if pred_t > bias_t

    mean_bias = sum of the bias across time/length of data
    -------------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and mean bias
    """

    joined = (
        data.join(pred, on="date", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(diff=(pl.col("data") - pl.col("pred")))
    )

    joined = joined.with_columns(bias=joined["diff"].sign())

    m_bias = pl.DataFrame(
        {
            "forecast_start": joined["date"].min(),
            "forecast_end": joined["date"].max(),
            "mbias": joined["bias"].sum() / joined.shape[0],
        }
    )

    return m_bias


def get_mean_bias(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Wrapper function of check_date_match and calculate_mean_bias
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and mean bias

    """
    # Check the conditions for date match:
    check_date_match(data, pred)
    m_bias = calculate_mean_bias(data, pred)

    return m_bias


def calculate_eos_abe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate the absolute error of the total uptake at the end of season between data and prediction
    relative to data.
    -------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and absolute error
        in the total uptake between data and prediction relative to data forecast end date.

    """
    joined = (
        data.join(pred, on="date", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(
            cumu_data=pl.col("data").cum_sum(),
            cumu_pred=pl.col("pred").cum_sum(),
        )
        .filter(pl.col("date") == pl.col("date").max())
        .rename({"date": "forecast_end"})
    )

    abe_prop = abs(joined["cumu_data"] - joined["cumu_pred"]) / joined["cumu_data"]

    eos_abe = pl.DataFrame(
        {
            "forecast_start": joined["date"].min(),
            "forecast_end": joined["date"].max(),
            "ae_prop": abe_prop,
        }
    )

    return eos_abe


def get_eos_abe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Wrapper function of check_date_match and calculate_eos_abe
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be IncidentUptakeData
    pred:
        The forecast made by model. Should be PointForecast

    Return
    pl.DataFrame with forecast start date, forecast end date, and absolute error
        in the total uptake between data and prediction relative to data forecast end date.

    """
    # Check the conditions for date match:
    check_date_match(data, pred)
    eos_abe = calculate_eos_abe(data, pred)

    return eos_abe
