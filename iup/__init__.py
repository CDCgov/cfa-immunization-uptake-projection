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
        Must have date and estimate columns; can have more
        """
        self.assert_in_schema({"date": pl.Date, "estimate": pl.Float64})

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
                .sort("date")
                .filter(pl.col("date") < start_date)
            )
        elif side == "test":
            out = (
                pl.concat(uptake_data_list)
                .sort("date")
                .filter(pl.col("date") >= start_date)
            )
        else:
            raise RuntimeError(f"Unrecognized side '{side}'")

        return out


class IncidentUptakeData(UptakeData):
    def to_cumulative(
        self, group_cols: tuple[str,] | None, last_cumulative=None
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

        return IncidentUptakeData(out)


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
    group_cols: List[List[str]],
) -> tuple | None:
    """
    Note the column names for grouping factors across data sets.

    Parameters
    group_cols: [[str,]]
        List of lists of column names for grouping factors, for each data set

    Returns
        (str,)
        The column names of grouping factors common to all data sets

    Details
    Before returning a single tuple of the desired column names,
    check that they are identical for every data set.
    """

    assert all([len(g) == len(group_cols[0]) for g in group_cols])
    assert all([g == group_cols[0] for g in group_cols])
    group_names = tuple(group_cols[0])

    return group_names


class QuantileForecast(Data):
    """
    Class for forecast with quantiles.
    Save for future.
    """

    def validate(self):
        self.assert_in_schema(
            {"date": pl.Date, "quantile": pl.Float64, "estimate": pl.Float64}
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
            {"date": pl.Date, "sample_id": pl.Int64, "estimate": pl.Float64}
        )
