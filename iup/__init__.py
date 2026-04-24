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
        """Verify that columns of the expected types are present in the data frame.

        Args:
            names_types: Column names and types mapping.
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


class CoverageData(Data):
    def validate(self):
        """Must have time_end and estimate columns; can have more."""
        self.assert_in_schema({"time_end": pl.Date, "estimate": pl.Float64})


class IncidentCoverageData(CoverageData):
    def validate(self):
        super().validate()
        if not self["estimate"].is_between(-1.0, 1.0).all():
            bad_values = (
                self.filter(pl.col("estimate").is_between(-1.0, 1.0).not_())["estimate"]
                .unique()
                .to_list()
            )
            raise ValueError(
                f"Incident coverage `estimate` must be have values between -1 and +1. "
                f"Values included {bad_values}"
            )

    def to_cumulative(
        self, groups: List[str,] | None, prev_cumulative: pl.DataFrame | None = None
    ) -> "CumulativeCoverageData":
        """Convert incident to cumulative coverage data.

        Cumulative sum of incident coverage gives the cumulative coverage.
        Optionally, additional cumulative coverage from before the start of
        the incident data may be provided.
        Even if no groups are specified, the data must at least be grouped by season.

        Args:
            groups: Names of the columns of grouping factors, or None. If `None`, then
                data will be grouped by `"season"`.
            prev_cumulative: Cumulative coverage from before the start of the incident
                data, for each group, or None. If `None`, group by `"season"`.

        Returns:
            Cumulative coverage on each date in the input incident coverage data.
        """
        if groups is None:
            groups = ["season"]

        out = self.with_columns(estimate=pl.col("estimate").cum_sum().over(groups))

        if prev_cumulative is not None:
            out = out.join(prev_cumulative, on=groups)

            out = out.with_columns(
                estimate=pl.col("estimate") + pl.col("last_cumulative")
            ).drop("last_cumulative")

        return CumulativeCoverageData(out)


class CumulativeCoverageData(CoverageData):
    def validate(self):
        super().validate()
        assert self["estimate"].is_between(0.0, 1.0).all(), (
            "Cumulative coverage `estimate` must be a proportion"
        )

    def to_incident(self, groups: List[str,] | None) -> IncidentCoverageData:
        """Convert cumulative to incident coverage data.

        Because the first report date for each group is often rollout,
        incident coverage on the first report date is 0.

        Args:
            groups: Names of the columns of grouping factors, or None. If `None`,
                then data will be grouped by `"season"`.

        Returns:
            Incident coverage on each date in the input cumulative coverage data.
        """
        if groups is None:
            groups = ["season"]

        out = self.with_columns(
            estimate=pl.col("estimate").diff().over(groups).fill_null(0)
        )

        return IncidentCoverageData(out)


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
            {"time_end": pl.Date, "sample_id": pl.UInt64, "estimate": pl.Float64}
        )


def date_to_season(
    date: pl.Expr, season_start_month: int, season_start_day: int = 1
) -> pl.Expr:
    """Extract the overwinter disease season from a date.

    Dates in year Y before the season start (e.g., Sep 1) are in the second part of
    the season (i.e., in season Y-1/Y). Dates in year Y after the season start are in
    season Y/Y+1. E.g., 2023-10-07 and 2024-04-18 are both in "2023/2024".

    Args:
        date: Dates in an coverage data frame.
        season_start_month: First month of the overwinter disease season.
        season_start_day: First day of the first month of the overwinter disease season.

    Returns:
        Seasons for each date.
    """

    # for every date, figure out the season breakpoint in that year
    season_start = pl.date(date.dt.year(), season_start_month, season_start_day)

    # what is the first year in the two-year season indicator?
    date_year = date.dt.year()
    year1 = pl.when(date < season_start).then(date_year - 1).otherwise(date_year)

    year2 = year1 + 1
    return pl.format("{}/{}", year1, year2)
