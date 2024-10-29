import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import abc
from typing_extensions import Self


class UptakeData(pl.DataFrame, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    @abc.abstractmethod
    def validate(self) -> None:
        pass

    def assert_columns_found(self, columns):
        for col in columns:
            assert col in self.columns, f"Column {col} is expected but not found."

    def assert_columns_type(self, columns, dtype):
        for col in columns:
            assert (
                self[col].dtype == dtype
            ), f"Column {col} should be {dtype} but is {self[col].dtype}"

    @staticmethod
    @abc.abstractmethod
    def split_train_test(uptake_data_list, start_date, side):
        pass


class IncidentUptakeData(UptakeData):
    def validate(self):
        self.assert_columns_found(
            ["region", "season", "date", "elapsed", "interval", "estimate"]
        )
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["region", "season"], pl.Utf8)
        self.assert_columns_type(["elapsed", "interval", "estimate"], pl.Float64)

    @staticmethod
    def split_train_test(uptake_data_list, start_date, side):
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

        out = IncidentUptakeData(out)

        return out

    def get_previous_uptake(self) -> Self:
        self = self.with_columns(previous=pl.col("estimate").shift(1).over("region"))

        return self

    def trim_outlier_intervals(self) -> Self:
        self = self.sort("date").with_row_count("index")
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

    def to_cumulative(self, last_cumulative):
        self = (
            self.select(["region", "date", "estimate"])
            .sort("date")
            .with_columns(estimate=pl.col("estimate").cum_sum().over("region"))
            .join(last_cumulative, on="region")
            .with_columns(estimate=pl.col("estimate") + pl.col("last_cumulative"))
            .drop("last_cumulative")
        )

        self = CumulativeUptakeData(self)

        return self


class CumulativeUptakeData(UptakeData):
    def validate(self):
        self.assert_columns_found(["region", "date", "estimate"])
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["region"], pl.Utf8)
        self.assert_columns_type(["estimate"], pl.Float64)
        assert (
            self["estimate"] >= 0.0
        ).all(), "Cumulative uptake estimates cannot be negative."

    def to_incident(self) -> IncidentUptakeData:
        self = (
            self.with_columns(
                season=(
                    pl.col("date").dt.year()
                    + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0)
                ).cast(pl.Utf8),
                estimate=pl.col("estimate").diff().over("region"),
                elapsed=(pl.col("date") - pl.col("date").first())
                .over("region")
                .dt.total_days()
                .cast(pl.Float64),
            )
            .with_columns(
                estimate=pl.col("estimate")
                .fill_null(pl.col("estimate").first())
                .over("region"),
                interval=pl.col("elapsed").diff().over("region"),
            )
            .with_columns(
                interval=pl.col("interval")
                .fill_null(pl.col("elapsed").first())
                .over("region")
            )
            .with_columns(estimate=(pl.col("estimate") / pl.col("interval")))
        )

        self = IncidentUptakeData(self)

        return self

    @staticmethod
    def split_train_test(uptake_data_list, start_date, side):
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

        out = CumulativeUptakeData(out)

        return out


def get_nis(
    data_path,
    region_col,
    date_col,
    estimate_col,
    rollout,
    filters=None,
) -> CumulativeUptakeData:
    out = (
        (pl.read_csv(data_path))
        .with_columns(estimate=pl.col(estimate_col).cast(pl.Float64, strict=False))
        .drop_nulls(subset=["estimate"])
    )

    if filters is not None:
        filter_expr = pl.lit(True)
        for k, v in filters.items():
            filter_expr &= pl.col(k).is_in(pl.lit(v))
        out = out.filter(filter_expr)

    out = (
        out.with_columns(
            region=pl.col(region_col),
            date=pl.col(date_col).str.to_date(
                "%m/%d/%Y %H:%M"
            ),  # "%m/%d/%Y %I:%M:%S %p"
        )
        .select(["region", "date", "estimate"])
        .sort("region", "date")
    )

    unique_regions = out["region"].unique()
    rollout_rows = pl.DataFrame(
        {
            "region": unique_regions,
            "date": [rollout for _ in range(len(unique_regions))],
            "estimate": [0.0 for _ in range(len(unique_regions))],
        }
    )
    out = out.vstack(rollout_rows).unique(maintain_order=True).sort("date")

    out = CumulativeUptakeData(out)  # rollout.str.to_datetime("%Y-%m-%d"))

    return out


def standardize(x, mn=None, sd=None):
    if mn is not None:
        return (x - mn) / sd
    else:
        return (x - x.mean()) / x.std()


def unstandardize(x, mn, sd):
    return x * sd + mn


class UptakeModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: UptakeData) -> Self:
        pass

    @abc.abstractmethod
    def predict(self, data: UptakeData, *args, **kwargs) -> UptakeData:
        pass


class LinearIncidentUptakeModel(UptakeModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, data: IncidentUptakeData) -> Self:
        self.start = data.group_by("region").agg(
            [
                pl.col("estimate").last().alias("last_estimate"),
                pl.col("date").last().alias("last_date"),
                pl.col("elapsed").last().alias("last_elapsed"),
                (pl.col("estimate") * pl.col("interval"))
                .filter(pl.col("season") == pl.col("season").max())
                .sum()
                .alias("last_cumulative"),
            ]
        )

        data = data.get_previous_uptake()

        data = IncidentUptakeData(  # Not sure why I must reinforce type here
            data.trim_outlier_intervals().with_columns(
                previous_std=pl.col("previous").pipe(standardize),
                elapsed_std=pl.col("elapsed").pipe(standardize),
                estimate_std=pl.col("estimate").pipe(standardize),
            )
        )

        self.previous_mean = data["previous"].mean()
        self.previous_sdev = data["previous"].std()
        self.elapsed_mean = data["elapsed"].mean()
        self.elapsed_sdev = data["elapsed"].std()
        self.estimate_mean = data["estimate"].mean()
        self.estimate_sdev = data["estimate"].std()

        self.x = (
            data.select(["previous_std", "elapsed_std"])
            .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
            .to_numpy()
        )

        self.y = data.select(["estimate_std"]).to_numpy()

        self.model.fit(self.x, self.y)

        return self

    def predict(
        self,
        start_date,
        end_date,
        interval,
    ) -> Self:
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
                estimate=pl.lit(0),
                previous=pl.lit(0),
            )
            .drop(["last_elapsed", "last_interval"])
        )

        regions = self.incident_projection["region"].value_counts()

        for r in range(regions.shape[0]):
            proj = np.zeros((regions["count"][r]) + 1)
            proj[0] = self.start.filter(pl.col("region") == regions["region"][r])[
                "last_estimate"
            ][0]

            for i in range(proj.shape[0] - 1):
                x = np.column_stack(
                    (
                        standardize(proj[i], self.previous_mean, self.previous_sdev),
                        standardize(
                            self.incident_projection["elapsed"][i],
                            self.elapsed_mean,
                            self.elapsed_sdev,
                        ),
                    )
                )
                x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
                y = self.model.predict(x)
                proj[i + 1] = unstandardize(
                    y[(0, 0)], self.estimate_mean, self.estimate_sdev
                )

            self.incident_projection = self.incident_projection.with_columns(
                estimate=pl.when(pl.col("region") == self.start["region"][r])
                .then(pl.Series(np.delete(proj, 0)))
                .otherwise(pl.col("estimate"))
            )

        self.incident_projection = self.incident_projection.with_columns(
            estimate=pl.col("estimate") * pl.col("interval")
        ).with_columns(previous=pl.col("estimate").shift(1).over("region"))

        self.incident_projection = IncidentUptakeData(self.incident_projection)
        self.cumulative_projections = self.incident_projection.to_cumulative(
            self.start.select(["region", "last_cumulative"])
        )

        return self
