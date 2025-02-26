import polars as pl
import pytest


@pytest.fixture
def frame():
    """
    Make a mock data frame to test data cleaning.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA", "PA", "USA"],
            "time_end": ["2020-01-07", "2020-01-14", "2020-01-21"],
            "estimate": [0.0, 0.1, 0.2],
            "indicator": ["refused", "booster", "booster"],
            "season": ["2019/2020", "2019/2020", "2019/2020"],
        }
    )

    return frame
