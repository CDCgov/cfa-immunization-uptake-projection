import datetime as dt
from typing import List

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
    def split_train_test(cls, uptake_data: "UptakeData", split_date: dt.date) -> tuple:
        """
        Subset a training or test set from data.

        Parameters
        uptake_data: UptakeData
            cumulative or incident uptake data
        split_date: dt.date
            date at which to split data

        Returns
        pl.DataFrames
            training and test portions of the cumulative or uptake data

        Details
        Training data are before the start date; test data are on or after.
        Infers what type of UptakeData to return from what type was given.
        """
        train = uptake_data.sort("time_end").filter(pl.col("time_end") < split_date)
        test = uptake_data.sort("time_end").filter(pl.col("time_end") >= split_date)

        return type(uptake_data)(train), type(uptake_data)(test)


class IncidentUptakeData(UptakeData):
    def validate(self):
        super().validate()
        if not self["estimate"].is_between(-1.0, 1.0).all():
            bad_values = (
                self.filter(pl.col("estimate").is_between(-1.0, 1.0).not_())["estimate"]
                .unique()
                .to_list()
            )
            raise ValueError(
                f"Incident uptake `estimate` must be have values between -1 and +1. "
                f"Values included {bad_values}"
            )

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


class CumulativeUptakeData(UptakeData):
    def validate(self):
        super().validate()
        assert self["estimate"].is_between(0.0, 1.0).all(), (
            "Cumulative uptake `estimate` must be a proportion"
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


class QuantileForecast(Data):
    """
    Class for forecast with quantiles.
    Save for future.
    """

    def validate(self):
        self.assert_in_schema(
            {"time_end": pl.Date, "quantile": pl.Float64, "estimate": pl.Float64}
        )

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
        super().validate()
        assert (self["quantile"] == 0.50).all()


class SampleForecast(Data):
    """
    Class for forecast with posterior distribution.
    Save for future.
    """

    def validate(self):
        self.assert_in_schema(
            {"time_end": pl.Date, "sample_id": pl.String, "estimate": pl.Float64}
        )
