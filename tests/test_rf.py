import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor

import vcf


@pytest.fixture
def frame():
    """
    Make a mock data frame to test model building.
    """
    frame = pl.DataFrame(
        {
            "geography": ["USA"] * 6,
            "time_end": [
                "2018-09-01",
                "2018-10-01",
                "2018-11-01",
                "2019-09-01",
                "2019-10-01",
                "2019-11-01",
            ],
            "estimate": np.linspace(1e-4, 5e-4, num=6),
            "lci": np.linspace(0.5e-4, 2.5e-4, num=6),
            "uci": np.linspace(2e-4, 1e-3, num=6),
            "sample_size": [1000] * 6,
            "season": ["2018/2019"] * 3 + ["2019/2020"] * 3,
        },
        schema_overrides={"time_end": pl.Date},
    )

    return frame


@pytest.fixture
def rf(frame):
    return vcf.RFModel(
        data=frame,
        season={"start_month": 9, "start_day": 1, "end_month": 4, "end_day": 1},
        params={"n_estimators": 10},
        forecast_date=datetime.date(2019, 10, 1),
        quantiles=[0.025, 0.5, 0.975],
    ).fit()


def test_fit(rf):
    assert isinstance(rf.model, RandomForestRegressor)


def test_predict(rf):
    pred = rf.predict()

    assert isinstance(pred, pl.DataFrame)
    assert set(pred.columns) == {
        "season",
        "geography",
        "time_end",
        "forecast_date",
        "quantile",
        "estimate",
    }
    assert pred.shape[0] > 0  # should have at least 1 row
