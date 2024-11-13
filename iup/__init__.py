import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import abc
from typing_extensions import Self
from typing import List


class UptakeData(pl.DataFrame, metaclass=abc.ABCMeta):
    """
    Abstract class for different forms of uptake data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an UptakeData object as a polars data frame plus validation.
        """
        super().__init__(*args, **kwargs)
        self.validate()
        self.sort("date")

    def validate(self):
        """
        Validate that an UptakeData object has the two key columns:
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

    def with_columns(self, *args, **kwargs):
        """
        Copy of polars with_columns that returns the same subclass as it's given.
        """
        orig_class = type(self)
        result = super().with_columns(*args, **kwargs)
        return orig_class(result)

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
        Concatenate UptakeData objects and split into training and test data.

        Parameters
        uptake_data_list: List[UptakeData]
            cumulative or incident uptake data objects, often from different seasons
        start_date: dt.date
            the first date for which projections should be made
        side: str
            whether the "train" or "test" portion of the data is desired

        Returns
        UptakeData
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


class IncidentUptakeData(UptakeData):
    """
    Subclass of UptakeData for incident uptake.
    """

    def trim_outlier_intervals(
        self, group_cols: tuple[str,] | None, threshold: float = 1.0
    ):
        """
        Remove rows from incident uptake data with intervals that are too large.

        Parameters
          group_cols (tuple) | None: names of grouping factor columns
          threshold (float): maximum standardized interval between first two dates

        Returns
        IncidentUptakeData
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


class CumulativeUptakeData(UptakeData):
    """
    Subclass of UptakeData for cumulative uptake.
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
    group_cols: dict,
    date_format: str,
) -> pl.DataFrame:
    """
    Select the date, region, and uptake estimate columns of NIS uptake data.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    estimate_col: str
        name of the NIS column for the uptake estimate (population %)
    date_col: str
        name of the NIS column for the date
    group_cols: dict
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
    frame: pl.DataFrame, rollout: dt.date, group_cols: dict
) -> pl.DataFrame:
    """
    Insert into NIS uptake data rows with 0 uptake on the rollout date.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    rollout: dt.date
        rollout date
    group_cols: dict
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
):
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
        assert all(
            [
                g.get(v) == group_cols[0].get(v)
                for g, v in zip(group_cols, group_cols[0])
            ]
        )
        group_names = tuple([v for v in group_cols[0].values()])

    return group_names


def standardize(x, mn=None, sd=None):
    """
    Standardize: subtract mean and divide by standard deviation.

    Parameters
    x: pl.Expr | pl.DataFrame
        the numbers to standardize
    mn: float64
        the term to subtract, if not the mean of x
    sd: float64
        the term to divide by, if not the standard deviation of x

    Returns
    pl.Expr | pl.DataFrame
        the standardized numbers
    """
    if mn is not None:
        return (x - mn) / sd
    else:
        return (x - x.mean()) / x.std()


def unstandardize(x, mn, sd):
    """
    Unstandardize: add standard deviation and multiply by mean.

    Parameters
    x: pl.Expr | pl.DataFrame
        the numbers to unstandardize
    mn: float64
        the term to add
    sd: float64
        the term to multiply by

    Returns
    pl.Expr | pl.DataFrame
        the unstandardized numbers
    """
    return x * sd + mn


class UptakeModel(abc.ABC):
    """
    Abstract class for different types of models.
    """

    @abc.abstractmethod
    def fit(self, data: UptakeData) -> Self:
        pass

    @abc.abstractmethod
    def predict(self, data: UptakeData, *args, **kwargs) -> UptakeData:
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
            the model with projection starting conditions, standardization
            constants, predictor and outcome variables, and the model fit
            all stored as attributes

        Details
        Extra columns for fitting this model are added to the incident data:
        - the disease season that each date belongs to
        - the interval of time in days between each successive date
        - the number of days elapsed between rollout and each date
        - the daily-average uptake in the interval preceding each date
        - the daily-average uptake in the interval preceding the previous date

        The daily-average uptake is modeled (rather than the total uptake) to
        account for slight variations in interval lengths (e.g. 6 vs. 7 days).

        Some starting conditions must be recorded to enable prediction later:
        - the last incident uptake in the training data
        - the last date in the training data
        - the last days-elapsed-since-rollout in the training data
        - the cumulative uptake at the end of the training data
        These are recorded separately for each group in the training data.

        If the training data spans multiple (combinations of) groups,
        complete pooling will be used to recognize the groups as distinct but
        to assume they behave identically except for initial conditions.

        Standardization constants must also be recorded for the model's outcome
        and first-order predictors, to enable projection later.

        Finally, the model is fit using the scikit-learn module.
        """
        data = (
            data.with_columns(
                season=pl.col("date").pipe(UptakeData.date_to_season),
                elapsed=pl.col("date")
                .pipe(UptakeData.date_to_elapsed)
                .over(group_cols),
                interval=pl.col("date")
                .pipe(UptakeData.date_to_interval)
                .over(group_cols),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1).over(group_cols))
        )

        self.start = data.group_by(group_cols).agg(
            [
                pl.col("daily").last().alias("last_daily"),
                pl.col("date").last().alias("last_date"),
                pl.col("elapsed").last().alias("last_elapsed"),
                (pl.col("estimate"))
                .filter(pl.col("season") == pl.col("season").max())
                .sum()
                .alias("last_cumulative"),
            ]
        )

        data = data.trim_outlier_intervals(group_cols).with_columns(
            previous_std=pl.col("previous").pipe(standardize),
            elapsed_std=pl.col("elapsed").pipe(standardize),
            daily_std=pl.col("daily").pipe(standardize),
        )

        self.standards = {
            "previous": {
                "mean": data["previous"].mean(),
                "std": data["previous"].std(),
            },
            "elapsed": {"mean": data["elapsed"].mean(), "std": data["elapsed"].std()},
            "daily": {"mean": data["daily"].mean(), "std": data["daily"].std()},
        }

        self.x = (
            data.select(["previous_std", "elapsed_std"])
            .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
            .to_numpy()
        )

        self.y = data.select(["daily_std"]).to_numpy()

        self.model.fit(self.x, self.y)

        return self

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
        desired time frame with the desired time intervals.

        The starting conditions derived from the last training data date
        are used to project for the first date. From there, projections
        are generated sequentially, because the projection for each date
        depends on the previous date, thanks to the model structure.

        Note the each projection must be unstandardized using the mean/std
        from the outcome variable in the training data, then restandardized
        using the mean/std from the "previous" predictor in the training data.

        The sequential projection loop does not yet incorporate uncertainty.
        This must be included in the future.

        After projections are completed, they are converted from daily-average
        to total incident uptake, as well as cumulative uptake, on each date.
        """
        self.start = self.start.with_columns(
            last_interval=(start_date - pl.col("last_date")).dt.total_days()
        )

        self.incident_projection = (
            pl.date_range(
                start=start_date,
                end=end_date,
                interval=interval,
                eager=True,
            )
            .alias("date")
            .to_frame()
        )

        if group_cols is not None:
            self.incident_projection = self.incident_projection.join(
                self.start.select(group_cols), how="cross"
            )

        self.incident_projection = self.incident_projection.with_columns(
            elapsed=((pl.col("date") - start_date).dt.total_days().cast(pl.Float64)),
            season=(
                pl.col("date").dt.year()
                + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0)
            ).cast(pl.Utf8),
        ).with_columns(interval=pl.col("elapsed").diff().over(group_cols))

        if group_cols is not None:
            self.incident_projection = self.incident_projection.join(
                self.start.select(list(group_cols) + ["last_elapsed", "last_interval"]),
                on=group_cols,
            )
        else:
            self.incident_projection = self.incident_projection.with_columns(
                last_elapsed=self.start["last_elapsed"][0],
                last_interval=self.start["last_interval"][0],
            )

        self.incident_projection = self.incident_projection.with_columns(
            elapsed=pl.col("elapsed")
            + pl.col("last_elapsed")
            + pl.col("last_interval"),
            interval=pl.when(pl.col("interval").is_null())
            .then(pl.col("last_interval"))
            .otherwise(pl.col("interval")),
            daily=pl.lit(0),
            estimate=pl.lit(0),
            previous=pl.lit(0),
        ).drop(["last_elapsed", "last_interval"])

        if group_cols is not None:
            groups = self.incident_projection.partition_by(group_cols)
        else:
            groups = [self.incident_projection]

        for g in range(len(groups)):
            proj = np.zeros(groups[g].shape[0] + 1)

            if group_cols is not None:
                proj[0] = self.start.join(groups[g], on=group_cols, how="semi")[
                    "last_daily"
                ][0]
            else:
                proj[0] = self.start["last_daily"][0]

            for i in range(proj.shape[0] - 1):
                x = np.column_stack(
                    (
                        standardize(
                            proj[i],
                            self.standards["previous"]["mean"],
                            self.standards["previous"]["std"],
                        ),
                        standardize(
                            groups[g]["elapsed"][i],
                            self.standards["elapsed"]["mean"],
                            self.standards["elapsed"]["mean"],
                        ),
                    )
                )
                x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
                y = self.model.predict(x)
                proj[i + 1] = unstandardize(
                    y[(0, 0)],
                    self.standards["daily"]["mean"],
                    self.standards["daily"]["std"],
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
