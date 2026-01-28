import polars as pl
import pytest

import iup
import iup.utils


@pytest.fixture
def frame(season_start_month=9, season_start_day=1):
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
                "N_vax": [1, 1, 100, 10, 300, 30, 400, 40],
                "estimate": [0.001, 0.001, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04],
                "season": "2019/2020",
            }
        )
        .with_columns(time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))
        .with_columns(
            elapsed=iup.utils.date_to_elapsed(
                pl.col("time_end"), season_start_month, season_start_day
            )
            / 365,
        )
    )

    frame = iup.CumulativeCoverageData(frame)

    return frame
