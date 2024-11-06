from iup import IncidentUptakeData
from datetime import date
from polars.testing import assert_frame_equal
import polars as pl


def test_inc_uptake_trim_outlier():
    """If all dates are equally spaced, drop the first two rows"""
    input_df = IncidentUptakeData(
        {
            "region": ["TX"] * 3 + ["CA"] * 3,
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)] * 2,
            "estimate": [0.1] * 6,
            "interval": "week",
        }
    ).with_columns(
        interval=IncidentUptakeData.date_to_interval(pl.col("date")),
        elapsed=IncidentUptakeData.date_to_elapsed(pl.col("date")),
    )

    df = input_df.trim_outlier_intervals()

    expected = input_df.filter(pl.col("elapsed") >= 2)

    assert_frame_equal(df, expected, check_row_order=False)
