import datetime

import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor

from iup.models import RFModel


@pytest.fixture
def rf(frame):
    return RFModel(
        data=frame,
        season={
            "start_month": 7,
            "start_day": 1,
            "end_month": 4,
            "end_day": 1,
            "end_year": 2020,
        },
        params={"n_estimators": 10},
        forecast_date=datetime.date(2019, 8, 1),
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
