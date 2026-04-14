import polars as pl
import pytest

import iup
import iup.utils


@pytest.fixture
def frame(season_start_month=9, season_start_day=1):
    """
    Make a mock data frame to test model building.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA"] * 20,
            "time_end": [
                "2018-07-01",
                "2018-08-01",
                "2018-09-01",
                "2018-10-01",
                "2018-11-01",
                "2018-12-01",
                "2019-01-01",
                "2019-02-01",
                "2019-03-01",
                "2019-04-01",
                "2019-07-01",
                "2019-08-01",
                "2019-09-01",
                "2019-10-01",
                "2019-11-01",
                "2019-12-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
            ],
            "lci": [0.0, 0.0004, 0.04, 0.004, 0.1, 0.01, 0.2, 0.01, 0.23, 0.30] * 2,
            "uci": [0.004, 0.005, 0.3, 0.02, 0.5, 0.05, 0.6, 0.06, 0.30, 0.50] * 2,
            "sample_size": [1000, 1000, 2000, 3000, 4000, 4000, 4000, 5000, 4000, 5000]
            * 2,
            "estimate": [0.001, 0.001, 0.1, 0.01, 0.3, 0.03, 0.4, 0.04, 0.25, 0.40] * 2,
            "season": ["2018/2019"] * 10 + ["2019/2020"] * 10,
        },
        schema_overrides={"time_end": pl.Date},
    )

    frame = iup.CumulativeCoverageData(frame)

    return frame
