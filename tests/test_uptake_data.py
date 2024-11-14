import iup
import polars as pl
import pytest
import datetime as dt


@pytest.fixture
def frame():
    """
    Make a mock data frame to test data cleaning.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
            "date": [
                "2019-12-30",
                "2019-12-30",
                "2020-01-07",
                "2020-01-07",
                "2020-01-14",
                "2020-01-14",
                "2020-01-21",
                "2020-01-21",
            ],
            "estimate": [0.0, 0.0, 1.0, 0.1, 3.0, 0.3, 4.0, 0.4],
        }
    )

    frame = frame.with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

    return frame


def test_date_to_season(frame):
    """
    Return the overwinter season, for both fall and spring dates
    """
    output = frame.with_columns(date=iup.UptakeData.date_to_season(pl.col("date")))

    assert all(output["date"] == pl.Series(["2019/2020"] * 8))


def test_date_to_interval(frame):
    """
    Return the interval between dates by grouping factor
    """
    output = frame.with_columns(
        interval=iup.UptakeData.date_to_interval(pl.col("date")).over("geography")
    )

    assert all(
        output["interval"][2:8]
        == pl.Series(
            [
                8.0,
                8.0,
                7.0,
                7.0,
                7.0,
                7.0,
            ]
        )
    )


def test_date_to_elapsed(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.with_columns(
        elapsed=iup.UptakeData.date_to_elapsed(pl.col("date")).over("geography")
    )

    assert all(
        output["elapsed"]
        == pl.Series(
            [
                0.0,
                0.0,
                8.0,
                8.0,
                15.0,
                15.0,
                22.0,
                22.0,
            ]
        )
    )


def test_split_train_test_handles_train(frame):
    """
    Return the training half of a data set.
    """
    frame2 = frame.with_columns(date=pl.col("date") + pl.duration(days=365))
    start_date = dt.date(2020, 6, 1)

    output = iup.UptakeData.split_train_test([frame, frame2], start_date, "train")

    assert output.equals(frame)


def test_split_train_test_handles_test(frame):
    """
    Return the testing half of a data set.
    """
    frame2 = frame.with_columns(date=pl.col("date") + pl.duration(days=365))
    start_date = dt.date(2020, 6, 1)

    output = iup.UptakeData.split_train_test([frame, frame2], start_date, "test")

    assert output.equals(frame2)


def test_trim_outlier_intervals_handles_two_rows(frame):
    """
    If there are two or fewer rows (per group), all rows should be trimmed.
    """
    frame = iup.IncidentUptakeData(frame.filter(pl.col("date") < dt.date(2020, 1, 9)))

    output = frame.trim_outlier_intervals(group_cols=("geography",))

    assert output.shape[0] == 0


def test_trim_outlier_intervals_handles_above_threshold(frame):
    """
    If the first interval is too big, first three rows are trimmed by group.
    """
    frame = iup.IncidentUptakeData(frame)

    output = frame.trim_outlier_intervals(group_cols=("geography",))

    assert output.shape[0] == 2


def test_trim_outlier_intervals_handles_below_threshold(frame):
    """
    If the first interval is not too big, first two rows are trimmed by group.
    """
    frame = iup.IncidentUptakeData(frame)

    output = frame.trim_outlier_intervals(group_cols=("geography",), threshold=2)

    assert output.shape[0] == 4


def test_to_cumulative_handles_no_last(frame):
    """
    If last_cumulative is not given, then simple cumulative sums are performed
    """
    frame = iup.IncidentUptakeData(frame)

    output = frame.to_cumulative(group_cols=("geography",))

    assert all(
        output["estimate"]
        == pl.Series(
            [
                0.0,
                0.0,
                1.0,
                0.1,
                4.0,
                0.4,
                8.0,
                0.8,
            ]
        )
    )


def test_to_cumulative_handles_last(frame):
    """
    If last_cumulative is given, then cumulative sums are augmented
    """
    frame = iup.IncidentUptakeData(frame)

    last_cumulative = pl.DataFrame(
        {"last_cumulative": [1.0, 0.1], "geography": ["USA", "PA"]}
    )

    output = frame.to_cumulative(
        group_cols=("geography",), last_cumulative=last_cumulative
    )

    assert all(
        output["estimate"]
        == pl.Series(
            [
                1.0,
                0.1,
                2.0,
                0.2,
                5.0,
                0.5,
                9.0,
                0.9,
            ]
        )
    )


def test_to_cumulative_handles_no_groups(frame):
    """
    If there are no groups, cumulative sums are taken over the whole frame at once.
    """
    frame = iup.IncidentUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = frame.to_cumulative(group_cols=None)

    assert all(output["estimate"] == pl.Series([0.0, 1.0, 4.0, 8.0]))


def test_to_incident_handles_groups(frame):
    """
    If there are groups, successive differences are taken over the groups.
    """
    frame = iup.CumulativeUptakeData(frame)

    output = frame.to_incident(group_cols=("geography",))

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.0,
                0.0,
                1.0,
                0.1,
                2.0,
                0.2,
                1.0,
                0.1,
            ]
        )
    )


def test_to_incident_handles_no_groups(frame):
    """
    If there are no groups, successive differences are taken over the entire data frame.
    """
    frame = iup.CumulativeUptakeData(
        frame.filter(pl.col("geography") == "USA").drop("geography")
    )

    output = frame.to_incident(group_cols=None)

    assert all(
        output["estimate"].round(10)
        == pl.Series(
            [
                0.0,
                1.0,
                2.0,
                1.0,
            ]
        )
    )
