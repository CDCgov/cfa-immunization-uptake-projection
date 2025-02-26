import polars as pl
import pytest

import iup
import iup.models


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = (
        pl.DataFrame(
            {
                "geography": ["USA", "PA", "USA", "PA", "USA", "PA", "USA", "PA"],
                "time_end": [
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
                "season": "2019/2020",
                "elapsed": [0, 0, 7, 7, 14, 14, 21, 21],
                "interval": [None, None, 7, 7, 7, 7, 7, 7],
            }
        )
        .with_columns(daily=(pl.col("estimate") / pl.col("interval")).fill_null(0))
        .with_columns(previous=pl.col("daily").shift(1).over("geography"))
        .with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))
    )

    frame = iup.IncidentUptakeData(frame)

    return frame


def test_extract_standards(frame):
    """
    Make a dictionary with as many keys as provided variables,
    and two values per sub-dictionary
    """
    output = iup.models.LinearIncidentUptakeModel.extract_standards(
        frame, ("estimate", "elapsed")
    )

    correct = {
        "estimate": {"mean": frame["estimate"].mean(), "std": frame["estimate"].std()},
        "elapsed": {"mean": frame["elapsed"].mean(), "std": frame["elapsed"].std()},
    }

    assert output == correct
