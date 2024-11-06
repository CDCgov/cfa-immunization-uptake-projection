from iup import IncidentUptakeData
from datetime import date
from polars.testing import assert_frame_equal
import polars as pl


def test_inc_uptake_minimum_4_data_points():
    """If there are only 3 data points, drop all of them"""
    assert (
        IncidentUptakeData(
            {
                "region": "TX",
                "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                "estimate": 0.0,
                "interval": "BOGUS",
            }
        )
        .trim_outlier_intervals()
        .shape[0]
        == 0
    )


def test_inc_uptake_trim_outlier2():
    """If all dates are equally spaced, drop the first two rows"""
    dates = [
        date(2020, 1, 1),
        date(2020, 1, 2),
        date(2020, 1, 3),
        # note that dates cannot be exactly evenly spaced, because then
        # standardization ends up with zero SD in the denominator
        date(2020, 1, 5),
    ]

    input_df = IncidentUptakeData(
        {
            "region": ["TX"] * len(dates) + ["CA"] * len(dates),
            "date": dates * 2,
            "estimate": 0.0,
            "interval": "BOGUS",
        }
    )

    grouping_vars = ("region",)
    df = input_df.trim_outlier_intervals(grouping_vars)

    # we should have dropped two dates per region
    assert df.shape[0] == (len(dates) - 2) * 2

    # check the actual values
    expected = input_df.filter(pl.col("date") >= date(2020, 1, 3))
    assert_frame_equal(df, expected, check_row_order=False)


def test_inc_uptake_trim_outlier_other_groups():
    """If all dates are equally spaced, drop the first two rows, but
    now also show that we can have more groupings"""
    dates = [
        date(2020, 1, 1),
        date(2020, 1, 2),
        date(2020, 1, 3),
        # note that dates cannot be exactly evenly spaced, because then
        # standardization ends up with zero SD in the denominator
        date(2020, 1, 5),
    ]

    input_df = IncidentUptakeData(
        pl.DataFrame({"date": dates, "estimate": 0.0, "interval": "BOGUS"})
        .join(pl.DataFrame({"region": ["TX", "CA"]}), how="cross")
        .join(
            pl.DataFrame({"age": ["infant", "child", "adult", "older_adult"]}),
            how="cross",
        )
    )

    grouping_vars = ("region", "age")
    df = input_df.trim_outlier_intervals(grouping_vars)

    # we should have dropped two dates per region (x2) and age (x4)
    assert df.shape[0] == (len(dates) - 2) * 2 * 4

    # check the actual values
    expected = input_df.filter(pl.col("date") >= date(2020, 1, 3))
    assert_frame_equal(df, expected, check_row_order=False)


def test_inc_uptake_trim_outlier3():
    """If the first two dates are widely spaced, drop the first three rows"""
    dates = [
        date(2020, 1, 1),
        # note the big jump from Jan 1 to Feb 1
        date(2020, 2, 1),
        date(2020, 2, 2),
        date(2020, 2, 3),
    ]
    input_df = IncidentUptakeData(
        {
            "region": ["TX"] * len(dates) + ["CA"] * len(dates),
            "date": dates * 2,
            "estimate": 0.0,
            "interval": "BOGUS",
        }
    )

    df = input_df.trim_outlier_intervals()

    # we should have lost 3 dates per region
    assert df.shape[0] == (len(dates) - 3) * 2

    # check the actual values
    expected = input_df.filter(pl.col("date") >= date(2020, 2, 3))
    assert_frame_equal(df, expected, check_row_order=False)
