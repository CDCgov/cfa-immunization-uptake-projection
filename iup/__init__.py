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

    def validate(self):
        """
        Validate that an UptakeData object has the three key columns:
        date, grouping factors, and uptake estimate (% of population).
        """
        self.assert_columns_found(["date", "region", "estimate"])
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["region"], pl.Utf8)
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

    def trim_outlier_intervals(self) -> Self:
        """
        Remove rows from incident uptake data with intervals that are too large.

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
        self = self.sort("date")
        unique_regions = self["region"].unique()
        sub_frames = []

        for x in range(len(unique_regions)):
            this_region = pl.col("region") == unique_regions[x]
            if (standardize(self.filter(this_region)["interval"]))[1] > 1:
                sub_frames.append(self.filter(this_region).slice(3))
            else:
                sub_frames.append(self.filter(this_region).slice(2))

        self = pl.concat(sub_frames).sort("date")

        return self

    def to_cumulative(self, last_cumulative=None):
        """
        Convert incident to cumulative uptake data.

        Parameters
        last_cumulative: pl.DataFrame
            additional cumulative uptake for each region absent from the incident data

        Returns
        CumulativeUptakeData
            cumulative uptake on each date in the input incident uptake data

        Details
        When building models, the first 2-3 rows of incident uptake may be removed,
        such that the sum of incident does not reflect the entire cumulative uptake.
        To fix this, cumulative uptake from the removed rows may be supplied separately.
        """
        out = self.with_columns(estimate=pl.col("estimate").cum_sum().over("region"))

        if last_cumulative is not None:
            out = (
                out.join(last_cumulative, on="region")
                .with_columns(estimate=pl.col("estimate") + pl.col("last_cumulative"))
                .drop("last_cumulative")
            )

        out = CumulativeUptakeData(out)

        return out


class CumulativeUptakeData(UptakeData):
    """
    Subclass of UptakeData for cumulative uptake.
    """

    def to_incident(self) -> IncidentUptakeData:
        """
        Convert cumulative to incident uptake data.

        Returns
        IncidentUptakeData
            incident uptake on each date in the input cumulative uptake data

        Details
        Because the first date for each region is rollout, incident uptake is 0.
        """
        out = self.with_columns(
            estimate=pl.col("estimate").diff().over("region").fill_null(0)
        )

        out = IncidentUptakeData(out)

        return out


def parse_nis(
    path: str,
    region_col: str,
    date_col: str,
    estimate_col: str,
    date_format: str,
    rollout: dt.date,
    filters=None,
) -> CumulativeUptakeData:
    """
    Load and parse NIS cumulative uptake data from a source file or address.

    Parameters
    path: str
        file path or url address for NIS data to import
    region_col: str
        name of the NIS column for the geographic region
    date_col: str
        name of the NIS column for the date
    estimate_col: str
        name of the NIS column for the uptake estimate (population %)
    date_format: str
        format of the dates in the NIS date column
    rollout: dt.date
        date of rollout
    filters: dict
        filters to remove unnecessary rows from the NIS data
        keys are NIS column names and values are entries for rows to keep

    Returns
        CumulativeUptakeData
        NIS data parsed to just the cumulative uptake of interest

    Details
    Parsing includes the following steps:
    - import NIS data from source
    - apply filters if any are supplied
    - isolate the region, date, and estimate columns
    - insert an initial entry of 0 uptake on the rollout date
    """
    frame = fetch_nis(path)

    if filters is not None:
        frame = apply_filters(frame, filters)

    frame = select_columns(frame, region_col, date_col, estimate_col, date_format)

    frame = insert_rollout(frame, rollout)

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


def apply_filters(frame: pl.DataFrame, filters: dict) -> pl.DataFrame:
    """
    Apply filters to NIS data to remove unnecessary rows.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    filters: dict
        filters to remove unnecessary rows from the NIS data
        keys are NIS column names and values are entries for rows to keep

    Returns
    pl.DataFrame
        NIS cumulative uptake data with unnecessary rows removed

    Details
        If multiple entries in a column indicate that a row should be kept,
        the dictionary value for that column may be a list of these entries.
    """
    filter_expr = pl.lit(True)
    for k, v in filters.items():
        filter_expr &= pl.col(k).is_in(pl.lit(v))
    frame = frame.filter(filter_expr)

    return frame


def select_columns(
    frame: pl.DataFrame,
    region_col: str,
    date_col: str,
    estimate_col: str,
    date_format: str,
) -> pl.DataFrame:
    """
    Select the date, region, and uptake estimate columns of NIS uptake data.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    region_col: str
        name of the NIS column for the geographic region
    date_col: str
        name of the NIS column for the date
    estimate_col: str
        name of the NIS column for the uptake estimate (population %)
    date_format: str
        format of the dates in the NIS date column

    Returns
        NIS cumulative uptake data with only the necessary columns

    Details
        Only the date, region, and uptake estimate are necessary for
        cumulative uptake data. Region is a stand-in for any grouping factor.
    """
    frame = (
        frame.with_columns(
            estimate=pl.col(estimate_col).cast(pl.Float64, strict=False),
            region=pl.col(region_col),
            date=pl.col(date_col).str.to_date(date_format),
        )
        .drop_nulls(subset=["estimate"])
        .select(["region", "date", "estimate"])
        .sort("region", "date")
    )

    return frame


def insert_rollout(frame: pl.DataFrame, rollout: dt.date) -> pl.DataFrame:
    """
    Insert into NIS uptake data rows with 0 uptake on the rollout date.

    Parameters
    frame: pl.DataFrame
        NIS data in the midst of parsing
    rollout: dt.date
        rollout date

    Returns
        NIS cumulative data with rollout rows included

    Details
    A separate rollout row is added for every region.
    """
    unique_regions = frame["region"].unique()
    rollout_rows = pl.DataFrame(
        {
            "region": unique_regions,
            "date": rollout,
            "estimate": 0.0,
        }
    )
    frame = frame.vstack(rollout_rows).unique(maintain_order=True).sort("date")

    return frame


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

    Let t_i be increasing time points and UR_i be the uptake rate (e.g., uptake
    per day) between times t_i-1 and t_i (i.e., uptake up to time t_i). Then the
    model is:

    UR_i = beta0 + beta1*UR_i-1 + beta2*t_i + beta3*UR_i-1*t_i + error
    """

    def __init__(self):
        """
        Initialize the model as a scikit-learn linear regression.
        """
        self.model = LinearRegression()

    @staticmethod
    def _augment_df(df: pl.DataFrame) -> pl.DataFrame:
        """Add columns to data"""
        return (
            df.with_columns(
                season=pl.col("date").pipe(UptakeData.date_to_season),
                elapsed=pl.col("date").pipe(UptakeData.date_to_elapsed),
                interval=pl.col("date").pipe(UptakeData.date_to_interval),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1))
        )

    def fit(self, data: IncidentUptakeData, group_vars=("region",)) -> Self:
        """
        Fit a linear incident uptake model on training data.

        Parameters
        data: IncidentUptakeData
            training data on which to fit the model
        group_vars: sequence of strings
            columns in `data` representing separate times or populations

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
        These are recorded separately for each region in the training data.

        If the training data spans seasons, regions, or other grouping factors,
        complete pooling will be used to recognize the groups as distinct but
        to assume they behave identically except for initial conditions.

        Standardization constants must also be recorded for the model's outcome
        and first-order predictors, to enable projection later.

        Finally, the model is fit using the scikit-learn module.
        """
        assert set(group_vars).issubset(data.columns)
        self.group_vars = group_vars

        data = data.group_by(group_vars).map_groups(self._augment_df)

        self.start = data.group_by(group_vars).agg(
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

        data = (
            data.group_by(group_vars)
            .map_groups(lambda df: UptakeData(df).trim_outlier_intervals())
            .with_columns(
                previous_std=pl.col("previous").pipe(standardize),
                elapsed_std=pl.col("elapsed").pipe(standardize),
                daily_std=pl.col("daily").pipe(standardize),
            )
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

        # linear model is fit on inputs:
        # 1. previous uptake rate ("last daily")
        # 2. time from rollout to end of this predicted uptake period ("elapsed")
        # 3. interaction between the two terms
        # and it returns: uptake rate ("daily") in this period
        self.model.fit(self.x, self.y)

        return self

    def predict(
        self,
        start_date: dt.date,
        end_date: dt.date,
        interval: str,
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
            (
                pl.date_range(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    eager=True,
                )
                .alias("date")
                .to_frame()
                .join(self.start.select(self.group_vars), how="cross")
                .with_columns(
                    elapsed=(
                        (pl.col("date") - start_date).dt.total_days().cast(pl.Float64)
                    ),
                    season=(
                        pl.col("date").dt.year()
                        + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0)
                    ).cast(pl.Utf8),
                )
                .with_columns(interval=pl.col("elapsed").diff().over(self.group_vars))
            )
            .join(
                self.start.select(self.group_vars + ["last_elapsed", "last_interval"]),
                on=self.group_vars,
            )
            .with_columns(
                elapsed=pl.col("elapsed")
                + pl.col("last_elapsed")
                + pl.col("last_interval"),
                interval=pl.when(pl.col("interval").is_null())
                .then(pl.col("last_interval"))
                .otherwise(pl.col("interval")),
                daily=pl.lit(0),
                estimate=pl.lit(0),
                previous=pl.lit(0),
            )
            .drop(["last_elapsed", "last_interval"])
        )

        # `regions` is a data frame, one row per region, with columns `region` and `count`
        regions = self.incident_projection["region"].value_counts()

        # `r` is an integer index over range of number of regions
        for r in range(regions.shape[0]):
            # start with zeros, equal to number of rows that had region `r` in the original
            # data, plus one (to undo trimming?)
            proj = np.zeros((regions["count"][r]) + 1)
            # first projected value (the bonus one) is equal to the last value in
            # the original data
            proj[0] = self.start.filter(pl.col("region") == regions["region"][r])[
                "last_daily"
            ][0]

            # iterate over index `i` for each place in the projection
            for i in range(proj.shape[0] - 1):
                # x has three rows
                # row 1 are "daily" values
                # row 2 are "elapsed" values
                # row 3 are interaction (daily*elapsed) values
                # initialize column 1 with:
                #  - row 1 is the "daily" value in the training data (does this assume
                #    that the predictions are all after the training period?)
                #  - row 2 is the time elapsed since the start date until the first date
                #    to be predicted
                #  - row 3 is the interaction
                x = np.column_stack(
                    (
                        # de-standardize the previous output value
                        standardize(
                            proj[i],
                            self.standards["previous"]["mean"],
                            self.standards["previous"]["std"],
                        ),
                        standardize(
                            self.incident_projection["elapsed"][i],
                            self.standards["elapsed"]["mean"],
                            self.standards["elapsed"]["mean"],
                        ),
                    )
                )
                x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
                # use the fitted linear model to predict the value for the next (i+1)
                # forecast value, and then de-standardize
                y = self.model.predict(x)
                proj[i + 1] = unstandardize(
                    y[(0, 0)],
                    self.standards["daily"]["mean"],
                    self.standards["daily"]["std"],
                )
                # in the first iteration, the "daily" value to be fitted is the last
                # value from the training data. in other iterations, it is the predicted
                # value from the iteration before.

            # after done iterating over the prediction times:
            # assign the rows in the output data frame that are for this region
            # but drop the last row(?) in `proj`, which was initialized with value
            # zero but not updated
            self.incident_projection = self.incident_projection.with_columns(
                daily=pl.when(pl.col("region") == self.start["region"][r])
                .then(pl.Series(np.delete(proj, 0)))
                .otherwise(pl.col("daily"))
            )

        # add estimate, which is the estimated uptake rate times the duration of the
        # uptake period
        self.incident_projection = self.incident_projection.with_columns(
            estimate=pl.col("daily") * pl.col("interval")
        )

        self.incident_projection = IncidentUptakeData(self.incident_projection)

        self.cumulative_projection = self.incident_projection.to_cumulative(
            self.start.select(self.group_vars + ["last_cumulative"])
        ).select(self.group_vars + ["date", "estimate"])

        return self
