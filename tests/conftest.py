import polars as pl
import pytest

import iup


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = pl.DataFrame(
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
            "N_vax": [1, 1, 100, 10, 300, 30, 400, 40],
            "N_tot": [1000] * 8,
            "estimate": [0.001, 0.001, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04],
            "season": "2019/2020",
        }
    ).with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))

    frame = iup.CumulativeUptakeData(frame)

    return frame
