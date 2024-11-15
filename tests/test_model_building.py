import iup
import polars as pl
import pytest
import datetime as dt


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = (
        pl.DataFrame(
            {
                "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
                "date": [
                    "2019-12-31",
                    "2019-12-31",
                    "2020-01-07",
                    "2020-01-07",
                    "2020-01-14",
                    "2020-01-14",
                    "2020-01-21",
                    "2020-01-21",
                ],
                "estimate": [0.0, 0.0, 1.0, 0.1, 3.0, 0.3, 4.0, 0.4],
                "season": ["2019/2020"] * 8,
                "elapsed": [0, 0, 7, 7, 14, 14, 21, 21],
                "interval": [None, None, 7, 7, 7, 7, 7, 7],
            }
        )
        .with_columns(daily=(pl.col("estimate") / pl.col("interval")).fill_null(0))
        .with_columns(previous=pl.col("daily").shift(1).over("geography"))
        .with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    )

    frame = iup.IncidentUptakeData(frame)

    return frame


def test_extract_starting_conditions(frame):
    """
    Use the last date for each grouping factor
    """
    output = iup.LinearIncidentUptakeModel.extract_starting_conditions(
        frame, ("geography",)
    )

    assert output.sort("geography").equals(
        pl.DataFrame(
            {
                "geography": ["USA", "PA"],
                "last_date": [dt.date(2020, 1, 21), dt.date(2020, 1, 21)],
                "last_daily": [4.0 / 7, 0.4 / 7],
                "last_elapsed": [21, 21],
                "last_cumulative": [8.0, 0.8],
            }
        ).sort("geography")
    )


def test_extract_standards(frame):
    """
    Make a dictionary with as many keys as provided variables,
    and two values per sub-dictionary
    """
    output = iup.LinearIncidentUptakeModel.extract_standards(
        frame, ("estimate", "elapsed")
    )

    assert len(output) == 2
    assert len(output["estimate"]) == 2


def test_fit(frame):
    """
    Model should fit a line to two points, giving a 'perfect' fit
    """

    output = iup.LinearIncidentUptakeModel()
    output = output.fit(frame, ("geography",))

    assert output.model.score(output.x, output.y) == 1.0
