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
                "lci": [0.0, 0.0004, 0.04, 0.004, 0.1, 0.01, 0.2, 0.01],
                "uci": [0.004, 0.005, 0.3, 0.02, 0.5, 0.05, 0.6, 0.06],
                "sample_size": [1000, 1000, 2000, 3000, 4000, 4000, 4000, 5000],
                "estimate": [0.001, 0.001, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04],
                "season": "2019/2020",
            },
            schema_overrides={"time_end": pl.Date},
        )
        # .with_columns(pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"))
        .with_columns(
            t=(
                pl.col("time_end") - pl.date(2019, season_start_month, season_start_day)
            ).dt.total_days()
        )
    )

    frame = iup.CumulativeCoverageData(frame)

    return frame
