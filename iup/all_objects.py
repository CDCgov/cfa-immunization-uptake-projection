import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import abc
from typing_extensions import Self


class UptakeData(pl.DataFrame, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def validate(self) -> None:
        pass

    def assert_columns_is_superset(self, columns: [str]):
        """Assert this dataframe's columns is a superset of some set

        Args:
            columns (str]): Set of columns this dataframe should include

        Raises:
            RuntimeError: if a column is not found
        """
        for col in columns:
            assert (
                col in self.columns
            ), f"Column {col} expected, not found among columns {self.columns}"

    def assert_columns_type(self, columns: [str], dtype: pl.DataType):
        for col in columns:
            assert (
                self[col].dtype == dtype
            ), f"Column {col} should be type {dtype}, instead is {self[col].dtype}"


class IncidentUptakeData(UptakeData):
    def validate(self):
        self.assert_columns_is_superset(
            ["region", "date", "season", "elapsed", "interval", "estimate", "previous"]
        )
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["region", "season"], pl.Utf8)
        self.assert_columns_type(
            ["elapsed", "interval", "estimate", "previous"], pl.Float64
        )


class IncidentUptakeProjection(UptakeData):
    def validate(self):
        assert set(self.columns).issuperset(
            {"region", "date", "interval", "estimate"}
        ), "At least one essential column is missing."
        assert self["date"].dtype.is_temporal(), "Column 'date' is not temporal."
        assert (
            self["region"].dtype == pl.Utf8
        ), "Column 'region' and/or 'season' is not a string."
        assert all(
            self[x].dtype.is_numeric() for x in ["interval", "estimate"]
        ), "Column 'elapsed', 'interval', 'estimate' and/or 'previous' is not numeric."


class CumulativeUptakeData(UptakeData):
    def validate(self):
        assert set(self.columns).issuperset(
            {"region", "date", "estimate"}
        ), "At least one essential column is missing."
        assert self["date"].dtype.is_temporal(), "Column 'date' is not temporal."
        assert self["region"].dtype == pl.Utf8, "Column 'region' is not a string."
        assert self["estimate"].dtype.is_numeric(), "Column 'estimate' is not numeric."
        assert (
            self["estimate"] >= 0.0
        ).all(), "Cumulative uptake estimates cannot be negative."

    def to_incident(self, rollout: dt.datetime) -> IncidentUptakeData:
        self = (
            self.with_columns(
                season=pl.col("date").dt.year()
                + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0),
                estimate=pl.col("estimate")
                .diff()
                .fill_null(pl.col("estimate").first()),
                elapsed=(pl.col("date") - rollout).cast(pl.Float64)
                / (1000 * 60 * 60 * 24),
            )
            .with_columns(
                interval=pl.col("elapsed").diff().fill_null(pl.col("elapsed").first()),
                season=pl.col("season").cast(pl.Utf8),
            )
            .with_columns(estimate=pl.col("estimate") / pl.col("interval"))
            .with_columns(previous=pl.col("estimate").shift(1))
        )

        if ((self["interval"] - self["interval"].mean()) / self["interval"].std())[
            0
        ] > 1:
            self = self.slice(2, self.height - 1)
        else:
            self = self.slice(1, self.height - 1)

        self = IncidentUptakeData(self)
        self.validate()

        return self


class CumulativeUptakeProjection(UptakeData):
    def validate(self):
        assert set(self.columns).issuperset(
            {"region", "date", "estimate"}
        ), "At least one essential column is missing."
        assert self["date"].dtype.is_temporal(), "Column 'date' is not temporal."
        assert self["region"].dtype == pl.Utf8, "Column 'region' is not a string."
        assert self["estimate"].dtype.is_numeric(), "Column 'estimate' is not numeric."


def get_nis(
    data_path, region_col, date_col, estimate_col, filters=None
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
            date=pl.col(date_col)
            .str.split(" ")
            .list.get(0)
            .str.strip_chars()
            .str.to_date(format="%m/%d/%Y"),
        )
        .select(["region", "date", "estimate"])
        .sort("region", "date")
    )

    out = CumulativeUptakeData(out)
    out.validate()

    return out


class ProjectionSettings:
    def __init__(self, *args, start_date, end_date, interval, rollout_dates):
        """
        Set up a projection scenario.

        Parameters
        ----------
        start_date : string
            First date for which uptake projections are desired, in %Y-%m-%d format
        end_date: string
            Last date for which uptake projections are desired, in %Y-%m-%d format
        interval: string
            Time interval between projections, as specified in polars.date_range()
        rollout_dates: tuple of strings
            Vaccine rollout date for each data set provided, in %Y-%m-%d format
        *args: CumulativeUptakeData
            Cumulative uptake data sets from past and/or present year

        Details
        -------

        This is intended to be the primary object created by the user, containing
        all the information necessary to curate the past/present data, fit a model,
        generate projections, and perform evaluation.

        Returns
        -------
        ProjectionSettings
            All info necessary to fit a model, generate projections, and evaluate.
        """

        self.start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        self.interval = interval

        if type(rollout_dates) is tuple:
            self.rollout_dates = tuple(
                dt.datetime.strptime(rollout_date, "%Y-%m-%d").date()
                for rollout_date in rollout_dates
            )
        else:
            self.rollout_dates = (
                dt.datetime.strptime(rollout_dates, "%Y-%m-%d").date(),
            )

        assert all(
            isinstance(arg, CumulativeUptakeData) for arg in args
        ), "One or more of the provided data frames is not cumulative uptake data."

        assert len(self.rollout_dates) == len(
            args
        ), "The number of rollout dates does not match the number of data sets."

        all_data = pl.concat(args).sort("date")

        self.init_cumulative = all_data.filter(pl.col("date") < self.start_date).select(
            pl.col("estimate").last()
        )

        self.test_data = CumulativeUptakeData(
            all_data.filter(pl.col("date") >= self.start_date)
        )

        train_data = []
        for arg, rollout_date in zip(args, self.rollout_dates):
            train_data.append(arg.to_incident(rollout_date))
        self.train_data = IncidentUptakeData(
            pl.concat(train_data).sort("date").filter(pl.col("date") < self.start_date)
        )

        assert (self.start_date.year - 1 * (self.start_date.month < 7)) == int(
            self.train_data["season"].tail(1)[0]
        ), "Start date is too close to the start of the season."

        self.init_interval = (
            self.start_date - self.train_data["date"].tail(1)[0]
        ).total_seconds() / (60 * 60 * 24)

        self.init_elapsed = self.train_data["elapsed"].tail(1)[0] + self.init_interval

        self.init_incident = self.train_data["estimate"].tail(1)[0]

        self.validate()

    def validate(self):
        assert isinstance(
            self.start_date, dt.date
        ), "Projection start date is not a date."
        assert isinstance(self.end_date, dt.date), "Projection end date is not a date."
        assert isinstance(self.interval, str), "Projection interval is not a string."

        assert all(
            isinstance(rollout_date, dt.date) for rollout_date in self.rollout_dates
        ), "Rollout dates contain a non-date item."

        assert isinstance(
            self.test_data, CumulativeUptakeData
        ), "Test data are not cumulative uptake data."
        assert isinstance(
            self.train_data, IncidentUptakeData
        ), "Training data are not incident uptake data."

    def build_model(self) -> Self:
        this_model = LinearIncidentUptakeModel()
        this_model = this_model.fit(self.train_data)
        this_model = this_model.predict(
            self.train_data,
            self.start_date,
            self.end_date,
            self.interval,
            self.init_elapsed,
            self.init_interval,
            self.init_incident,
            self.init_cumulative,
        )

        setattr(self, "model", this_model)
        return self


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
        pass

    def fit(self, train: IncidentUptakeData) -> Self:
        assert isinstance(train, IncidentUptakeData)
        train.validate()

        train_std = train.with_columns(
            previous_std=(pl.col("previous") - pl.mean("previous"))
            / pl.std("previous"),
            elapsed_std=(pl.col("elapsed") - pl.mean("elapsed")) / pl.std("elapsed"),
            estimate_std=(pl.col("estimate") - pl.mean("estimate"))
            / pl.std("estimate"),
        )

        x = (
            train_std.select(["previous_std", "elapsed_std"])
            .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
            .to_numpy()
        )
        y = train_std.select(["estimate_std"]).to_numpy()

        self.model.fit(x, y)

        return self

    def predict(
        self,
        train: IncidentUptakeData,
        start_date,
        end_date,
        interval,
        init_elapsed,
        init_interval,
        init_incident,
        init_cumulative,
    ) -> Self:
        previous_mn = train["previous"].mean()
        previous_sd = train["previous"].std()
        elapsed_mn = train["elapsed"].mean()
        elapsed_sd = train["elapsed"].std()
        estimate_mn = train["estimate"].mean()
        estimate_sd = train["estimate"].std()

        incident_proj = (
            pl.date_range(
                start=start_date,
                end=end_date,
                interval=interval,
                eager=True,
            )
            .alias("date")
            .to_frame()
            .with_columns(
                region=pl.lit(train.select(pl.col("region").unique())),
                elapsed=(
                    (pl.col("date") - start_date).cast(pl.Float64)
                    / (1000 * 60 * 60 * 24)
                )
                + init_elapsed,
                season=pl.col("date").dt.year()
                + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0),
            )
            .with_columns(
                interval=pl.col("elapsed").diff().fill_null(init_interval),
                season=pl.col("season").cast(pl.Utf8),
            )
        )

        proj = np.zeros((incident_proj.shape[0]))
        proj[0] = init_incident

        for i in range(proj.shape[0] - 1):
            x = np.column_stack(
                (
                    (proj[i] - previous_mn) / previous_sd,
                    (incident_proj["elapsed"][i + 1] - elapsed_mn) / elapsed_sd,
                )
            )
            x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
            y = self.model.predict(x)
            proj[i + 1] = y[(0, 0)] * estimate_sd + estimate_mn

        incident_proj = incident_proj.with_columns(estimate=pl.Series(proj))

        cumulative_proj = incident_proj.with_columns(
            estimate=pl.col("estimate") * pl.col("interval")
        ).with_columns(estimate=pl.col("estimate").cum_sum() + init_cumulative)

        incident_proj = IncidentUptakeProjection(
            incident_proj.select("region", "date", "estimate", "interval")
        )
        incident_proj.validate()

        cumulative_proj = CumulativeUptakeProjection(
            cumulative_proj.select("region", "date", "estimate")
        )
        cumulative_proj.validate()

        setattr(self, "incident_projections", incident_proj)
        setattr(self, "cumulative_projections", cumulative_proj)

        return self
