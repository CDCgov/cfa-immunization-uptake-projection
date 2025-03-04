import datetime as dt
from typing import List

import polars as pl
from polars.datatypes.classes import DataTypeClass

import iup.utils


class Data(pl.DataFrame):
    """
    Abstract class for observed data and forecast data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def assert_in_schema(self, names_types: dict[str, DataTypeClass]):
        """
        Verify that column of the expected types are present in the data frame.

        Parameters
        names_types (dict[str, pl.DataType]):
            Column names and types
        """
        for name, type_ in names_types.items():
            if name not in self.schema.names():
                raise RuntimeError(f"Column '{name}' not found")
            elif (
                name in self.schema.names() and (name, type_) not in self.schema.items()
            ):
                actual_type = self.schema.to_python()[name]
                raise RuntimeError(
                    f"Column '{name}' has type {actual_type}, not {type_}"
                )
            else:
                assert (name, type_) in self.schema.items()


class UptakeData(Data):
    def validate(self):
        """
        Must have time_end and estimate columns; can have more
        """
        self.assert_in_schema({"time_end": pl.Date, "estimate": pl.Float64})

    @classmethod
    def split_train_test(
        cls, uptake_data: "UptakeData", start_date: dt.date, side: str
    ) -> "UptakeData":
        """
        Subset a training or test set from data.

        Parameters
        uptake_data: UptakeData
            cumulative or incident uptake data
        start_date: dt.date
            first date for which forecasts should be made
        side: str
            whether the "train" or "test" portion of the data is desired

        Returns
        pl.DataFrame
            training or test portion of the cumulative or uptake data

        Details
        Training data are before the start date; test data are on or after.
        Infers what type of UptakeData to return from what type was given.
        """
        if side == "train":
            out = uptake_data.sort("time_end").filter(pl.col("time_end") < start_date)
        elif side == "test":
            out = uptake_data.sort("time_end").filter(pl.col("time_end") >= start_date)
        else:
            raise RuntimeError(f"Unrecognized side '{side}'")

        return type(uptake_data)(out)

    @staticmethod
    def date_to_season(
        date: pl.Expr, season_start_month: int = 9, season_start_day: int = 1
    ) -> pl.Expr:
        """
        Extract the overwinter disease season from a date

        Parameters
        date: pl.Expr
            dates in an uptake data frame
        season_start_month: int
            first month of the overwinter disease season
        season_start_day: int
            first day of the first month of the overwinter disease season

        Returns
        pl.Expr
            seasons for each date

        Details
        Dates in year Y before the season start (e.g., Sep 1) are in the second part of
        the season (i.e., in season Y-1/Y). Dates in year Y after the season start are in
        season Y/Y+1. E.g., 2023-10-07 and 2024-04-18 are both in "2023/2024"
        """

        # for every date, figure out the season breakpoint in that year
        season_start = pl.date(date.dt.year(), season_start_month, season_start_day)

        # what is the first year in the two-year season indicator?
        date_year = date.dt.year()
        year1 = pl.when(date < season_start).then(date_year - 1).otherwise(date_year)

        year2 = year1 + 1
        return pl.format("{}/{}", year1, year2)


class IncidentUptakeData(UptakeData):
    def to_cumulative(
        self, groups: List[str,] | None, prev_cumulative=None
    ) -> "CumulativeUptakeData":
        """
        Convert incident to cumulative uptake data.

        Parameters
        groups: List[str,] | None
            name(s) of the columns of grouping factors
        last_cumulative: pl.DataFrame
            cumulative from before the start of the incident data, for each group

        Returns
        CumulativeUptakeData
            cumulative uptake on each date in the input incident uptake data

        Details
        Cumulative sum of incident uptake gives the cumulative uptake.
        Optionally, additional cumulative uptake from before the start of
        the incident data may be provided.
        Even if no groups are specified, the data must at least be grouped by season.
        """
        if groups is None:
            groups = ["season"]

        out = self.with_columns(estimate=pl.col("estimate").cum_sum().over(groups))

        if prev_cumulative is not None:
            out = out.join(prev_cumulative, on=groups)

            out = out.with_columns(
                estimate=pl.col("estimate") + pl.col("last_cumulative")
            ).drop("last_cumulative")

        return CumulativeUptakeData(out)

    def trim_outlier_intervals(
        self,
        groups: List[str,] | None,
        threshold: float = 1.0,
    ) -> "IncidentUptakeData":
        """
        Remove rows from incident uptake data with intervals that are too large.

        Parameters
        groups (tuple) | None
            names of grouping factor columns
        threshold (float):
            maximum standardized interval between first two dates

        Returns
        pl.DataFrame:
            incident uptake data with the outlier rows removed

        Details
        The first row (index 0) may be rollout, so the second row (index 1)
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

        Even if no groups are specified, the data must at least be grouped by season.
        """
        assert self["time_end"].is_sorted(), (
            "Chronological sorting got broken during data augmentation!"
        )

        if groups is None:
            groups = ["season"]

        rank = pl.col("time_end").rank().over(groups)
        shifted_standard_interval = (
            pl.col("interval").pipe(iup.utils.standardize).shift(1).over(groups)
        )

        return IncidentUptakeData(
            self.filter(
                (rank >= 4) | ((rank == 3) & (shifted_standard_interval <= threshold))
            ).sort("time_end")
        )


class CumulativeUptakeData(UptakeData):
    def validate(self):
        # same validations as UptakeData
        super().validate()
        # and also require that uptake be a proportion
        assert self["estimate"].is_between(0.0, 1.0).all(), (
            "cumulative uptake `estimate` must be a proportion"
        )

    def to_incident(self, groups: List[str,] | None) -> IncidentUptakeData:
        """
        Convert cumulative to incident uptake data.

        Parameters
        groups: (str,) | None
            name(s) of the columns of grouping factors

        Returns
        IncidentUptakeData
            incident uptake on each date in the input cumulative uptake data

        Details
        Because the first report date for each group is often rollout,
        incident uptake on the first report date is 0.
        Even if no groups are specified, the data must at least be grouped by season.
        """
        if groups is None:
            groups = ["season"]

        out = self.with_columns(
            estimate=pl.col("estimate").diff().over(groups).fill_null(0)
        )

        return IncidentUptakeData(out)

    def insert_rollouts(
        self,
        rollouts: List[dt.date],
        groups: List[str] | None,
        season_start_month: int,
        season_start_day: int,
    ) -> "CumulativeUptakeData":
        """
        Insert into cumulative uptake data rows with 0 uptake on rollout dates.

        Parameters
        rollout: List[dt.date]
            list of rollout dates
        groups: tuple[str] | None
            names of grouping factor columns

        Returns
            cumulative uptake data with rollout rows included

        Details
        A separate rollout row is added for every grouping factor combination.
        Even if groups are specified, the data should not be grouped by season.
        """
        if groups is None:
            groups = []

        frame = self

        # do not use season as a grouping column to insert rollouts
        if len([g for g in groups if g != "season"]) > 0:
            rollout_rows = frame.select(groups)
            if "season" in rollout_rows.columns:
                rollout_rows = rollout_rows.drop("season")
            rollout_rows = rollout_rows.unique().join(
                pl.DataFrame({"time_end": rollouts}), how="cross"
            )
        else:
            rollout_rows = pl.DataFrame({"time_end": rollouts, "estimate": 0.0})

        # add season as a column only after making rollout rows
        rollout_rows = rollout_rows.with_columns(
            estimate=0.0,
            season=pl.col("time_end").pipe(
                UptakeData.date_to_season,
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
        )

        frame = frame.vstack(rollout_rows.select(frame.columns)).sort("time_end")

        return CumulativeUptakeData(frame)


class QuantileForecast(Data):
    """
    Class for forecast with quantiles.
    Save for future.
    """

    def validate(self):
        self.assert_in_schema(
            {"time_end": pl.Date, "quantile": pl.Float64, "estimate": pl.Float64}
        )

        # all quantiles should be between 0 and 1
        assert self["quantile"].is_between(0.0, 1.0).all(), (
            "quantiles must be between 0 and 1"
        )


class PointForecast(QuantileForecast):
    """
    Class for forecast with point estimate
    A subclass when quantile is 50%
    For now, enforce the "quantile50" to be "estimate"
    """

    def validate(self):
        # same validations as for QuantileForecast
        super().validate()
        # but additionally require that there be only one quantile value
        assert (self["quantile"] == 0.50).all()


class SampleForecast(Data):
    """
    Class for forecast with posterior distribution.
    Save for future.
    """

    def validate(self):
        self.assert_in_schema(
            {"time_end": pl.Date, "sample_id": pl.Int64, "estimate": pl.Float64}
        )
