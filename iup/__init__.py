import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import abc
from typing_extensions import Self


class UptakeData(pl.DataFrame, metaclass=abc.ABCMeta):
    """
    Abstract class for different forms of uptake data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an UptakeData object as a polars data frame,
        plus validation.
        """
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        """
        Validate that an UptakeData object has the three key columns:
        date, grouping factors, and an uptake estimate (% of population).
        """
        self.assert_columns_found(["date", "region", "estimate"])
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["region"], pl.Utf8)
        self.assert_columns_type(["estimate"], pl.Float64)

    def assert_columns_found(self, columns):
        """
        Verify that expected columns are found.
        """
        for col in columns:
            assert col in self.columns, f"Column {col} is expected but not found."

    def assert_columns_type(self, columns, dtype):
        """
        Verify that columns have the expected type.
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
        Input dates must be chronologically sorted. Get the number of days between each.
        """
        interval = date_col.diff().dt.total_days().cast(pl.Float64)

        return interval

    @staticmethod
    def date_to_elapsed(date_col: pl.Expr) -> pl.Expr:
        """
        Extract a time elapsed column from a date column, as polars expressions.
        Input dates must be chronologically sorted. Get the number of days since the first date.
        """
        elapsed = (date_col - date_col.first()).dt.total_days().cast(pl.Float64)

        return elapsed

    @staticmethod
    def split_train_test(uptake_data_list, start_date: dt.date, side: str):
        """
        Concatenate a list of UptakeData objects in chronological order.
        Then split it at the start date, when projections should begin.

        Dates before the start date are used for model training.
        Dates after the start date (if any) are used for testing vs. projections.
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
        Remove report dates with intervals that are too large,
        which can create problematic outliers.
        This occurs between rollout and the first report.

        The first date (index 0) is always rollout.
        The first report is the second date (index 1).

        If the first report's interval is 1+ std dev bigger than the avg interval,
        remove the first three dates (rollout, first report, and second report).
        (The second report is removed because its previous report is the first, so
        the impact of the first report being an outlier impacts the second report).

        If the first report is not an outlier, it still must be removed,
        because its previous report is 0 from rollout, which itself is an outlier.
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
        Convert incident uptake to cumulative via cumulative sums.
        Because the first few dates of incident uptake may have been removed,
        the cumulative sum may not reflect the entire cumulative uptake.
        To fix this, the last cumulative uptake before truncation may be supplied.
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
        Convert cumulative uptake to incident via successive differences.
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
    Load and parse NIS data from a source file or address.

    Filters can be applied to extract only relevant rows.
    Filters are in a dictionary, with column names as keys,
    and a list of the acceptable entries in those columns as values.

    The crucial columns (date, grouping factors, and uptake estimate)
    are selected and the remaining columns are dropped.

    An initial entry of 0 uptake on the report date is inserted.
    """
    frame = fetch_nis(path)

    if filters is not None:
        frame = apply_filters(frame, filters)

    frame = select_columns(frame, region_col, date_col, estimate_col, date_format)

    frame = insert_rollout(frame, rollout)

    frame = CumulativeUptakeData(frame)

    return frame


def fetch_nis(path) -> pl.DataFrame:
    """
    Get NIS data from its file path or address.
    """
    return pl.read_csv(path)


def apply_filters(frame: pl.DataFrame, filters: dict) -> pl.DataFrame:
    """
    Apply filters, specified via a dictionary.
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
    Select the date, grouping factor, and uptake estimate columns.
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
    Insert rows with 0 uptake on the rollout date(s).
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
    """
    if mn is not None:
        return (x - mn) / sd
    else:
        return (x - x.mean()) / x.std()


def unstandardize(x, mn, sd):
    """
    Unstandardize: add standard deviation and multiply by mean.
    """
    return x * sd + mn


class UptakeModel(abc.ABC):
    """
    Abstract class for different types of models,
    in anticipation of having more than one type of model.
    """

    @abc.abstractmethod
    def fit(self, data: UptakeData) -> Self:
        pass

    @abc.abstractmethod
    def predict(self, data: UptakeData, *args, **kwargs) -> UptakeData:
        pass


class LinearIncidentUptakeModel(UptakeModel):
    """
    Subclass of UptakeModel for a linear model that relates
    incident uptake to the absolute time elapsed since rollout,
    the incident uptake from the previous report, and their interaction.
    """

    def __init__(self):
        """
        Initialize the model as a scikit-learn linear regression.
        """
        self.model = LinearRegression()

    def fit(self, data: IncidentUptakeData) -> Self:
        """
        Fit a linear incident uptake model on training data.

        If the training data spans seasons, regions, or other grouping factors,
        complete pooling will be used to recognize the groups as distinct
        but to assume that they all behave identically except for initial conditions.

        Necessary columns for fitting this model are added to the incident data.

        To account for slight variations in the intervals between report dates
        (e.g. 6 vs. 7 vs. 8 days), the daily average incident uptake is modeled.

        Some starting conditions must be recorded to enable prediction later:
        - the last incident uptake in the training data
        - the last date in the training data
        - the last days-elapsed-since-rollout in the training data
        - the cumulative uptake at the end of the training data

        Standardization constants must also be recorded for the key model variables:
        - previous daily-avg incident uptake (predictor)
        - days-elapsed-since rollout (predictor)
        - daily-avg incident uptake (outcome) *NOT exactly the same as "previous"

        A linear model regresses daily-avg incident uptake from each report on
        - daily-avg incident uptake from the previous report
        - days-elapsed since rollout
        - interaction of the aforementioned predictors
        """
        data = (
            data.with_columns(
                season=pl.col("date").pipe(UptakeData.date_to_season),
                elapsed=pl.col("date").pipe(UptakeData.date_to_elapsed).over("region"),
                interval=pl.col("date")
                .pipe(UptakeData.date_to_interval)
                .over("region"),
            )
            .with_columns(daily=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("daily").shift(1).over("region"))
        )

        self.start = data.group_by("region").agg(
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

        data = data.trim_outlier_intervals().with_columns(
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
        start_date,
        end_date,
        interval,
    ) -> Self:
        """
        Make projections from a fit model, and store them as a model attribute.

        Record the interval between the training report date and
        the first projection date, to enable projection for this date.

        Set up a data frame to house the incident projections over the
        desired time frame with desired time intervals.

        INSERT EXPLANATION OF THE STEPWISE PROJECTION PROCESS.
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
                .join(self.start.select("region"), how="cross")
                .with_columns(
                    elapsed=(
                        (pl.col("date") - start_date).dt.total_days().cast(pl.Float64)
                    ),
                    season=(
                        pl.col("date").dt.year()
                        + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0)
                    ).cast(pl.Utf8),
                )
                .with_columns(interval=pl.col("elapsed").diff().over("region"))
            )
            .join(
                self.start.select(["region", "last_elapsed", "last_interval"]),
                on="region",
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

        regions = self.incident_projection["region"].value_counts()

        for r in range(regions.shape[0]):
            proj = np.zeros((regions["count"][r]) + 1)
            proj[0] = self.start.filter(pl.col("region") == regions["region"][r])[
                "last_daily"
            ][0]

            for i in range(proj.shape[0] - 1):
                x = np.column_stack(
                    (
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
                y = self.model.predict(x)
                proj[i + 1] = unstandardize(
                    y[(0, 0)],
                    self.standards["daily"]["mean"],
                    self.standards["daily"]["std"],
                )

            self.incident_projection = self.incident_projection.with_columns(
                daily=pl.when(pl.col("region") == self.start["region"][r])
                .then(pl.Series(np.delete(proj, 0)))
                .otherwise(pl.col("daily"))
            )

        self.incident_projection = self.incident_projection.with_columns(
            estimate=pl.col("daily") * pl.col("interval")
        )

        self.incident_projection = IncidentUptakeData(self.incident_projection)

        self.cumulative_projections = self.incident_projection.to_cumulative(
            self.start.select(["region", "last_cumulative"])
        ).select(["region", "date", "estimate"])

        return self
