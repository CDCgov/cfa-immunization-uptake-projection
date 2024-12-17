import datetime as dt
from typing import List, Sequence

import polars as pl
from polars.datatypes.classes import DataTypeClass


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
        """Verify that column of the expected types are present in the data frame

        Args:
            names_types (dict[str, pl.DataType]): Column names and types
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

    @staticmethod
    def split_train_test(
        uptake_data_list: Sequence[Data], start_date: dt.date, side: str
    ) -> pl.DataFrame:
        """
        Concatenate Data objects and split into training and test data.

        Parameters
        uptake_data_list: Sequence[Data]
            cumulative or incident uptake data objects, often from different seasons
        start_date: dt.date
            the first date for which projections should be made
        side: str
            whether the "train" or "test" portion of the data is desired

        Returns
        pl.DataFrame
            cumulative or uptake data object of the training or test portion

        Details
        Training data are before the start date; test data are on or after.
        """
        if side == "train":
            out = (
                pl.concat(uptake_data_list)
                .sort("time_end")
                .filter(pl.col("time_end") < start_date)
            )
        elif side == "test":
            out = (
                pl.concat(uptake_data_list)
                .sort("time_end")
                .filter(pl.col("time_end") >= start_date)
            )
        else:
            raise RuntimeError(f"Unrecognized side '{side}'")

        return out

    @staticmethod
    def date_to_season(
        date: pl.Expr, season_start_month: int = 9, season_start_day: int = 1
    ) -> pl.Expr:
        """
        Extract winter season from a date

        Dates in year Y before the season start (e.g., Sep 1) are in the second part of
        the season (i.e., in season Y-1/Y). Dates in year Y after the season start are in
        season Y/Y+1. E.g., 2023-10-07 and 2024-04-18 are both in "2023/24"

        Parameters
        date: pl.Expr
            dates
        season_start_month: int
            month of the year the season starts
        season_start_day: int
            day of season_start_month that the season starts

        Returns
        pl.Expr
            seasons for each date
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
        self, group_cols: List[str,] | None, last_cumulative=None
    ) -> pl.DataFrame:
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

        return out


class CumulativeUptakeData(UptakeData):
    def validate(self):
        # same validations as UptakeData
        super().validate()
        # and also require that uptake be a proportion
        assert (
            self["estimate"].is_between(0.0, 1.0).all()
        ), "cumulative uptake `estimate` must be a proportion"

    def to_incident(self, group_cols: List[str,] | None) -> IncidentUptakeData:
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

        return IncidentUptakeData(out)

    def insert_rollout(
        self, rollout: List[dt.date], group_cols: List[str] | None
    ) -> pl.DataFrame:
        """
        Insert into cumulative uptake data rows with 0 uptake on rollout dates.

        Parameters
        rollout: List[dt.date]
            list of rollout dates
        group_cols: tuple[str] | None
            names of grouping factor columns

        Returns
            cumulative uptake data with rollout rows included

        Details
        A separate rollout row is added for every grouping factor combination.
        """
        frame = self

        if group_cols is not None:
            rollout_rows = (
                frame.select(group_cols)
                .unique()
                .join(pl.DataFrame({"time_end": rollout}), how="cross")
                .with_columns(estimate=0.0)
            )
            group_cols = group_cols + ["season"]
        else:
            rollout_rows = pl.DataFrame({"time_end": rollout, "estimate": 0.0})
            group_cols = ["season"]

        frame = frame.vstack(rollout_rows.select(frame.columns)).sort("time_end")

        # Check that, after adding rollout, the first date for each group and season
        # is the one with the minimum estimate.
        assert (
            (
                frame.with_columns(
                    season=pl.col("time_end").pipe(UptakeData.date_to_season)
                )
                .with_columns(
                    min=(pl.col("estimate") - pl.min("estimate")).over(group_cols)
                )
                .group_by(group_cols)
                .first()
            )["min"]
            == 0.0
        ).all()

        return frame


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
        assert (
            self["quantile"].is_between(0.0, 1.0).all()
        ), "quantiles must be between 0 and 1"


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
