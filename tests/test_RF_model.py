import datetime

import polars as pl
from sklearn.ensemble import RandomForestRegressor

from iup.models import RFModel

PARAMS = {
    "start_month": 7,
    "start_day": 1,
    "end_month": 4,
    "end_day": 1,
    "end_year": 2020,
    "n_estimators": 100,
}

FORECAST_DATE = datetime.date(2019, 8, 1)
QUANTILES = [0.025, 0.5, 0.975]


class TestRFModel:
    def __init__(self, frame):
        self.rf = RFModel(
            data=frame,
            params=PARAMS,
            forecast_date=FORECAST_DATE,
            quantiles=QUANTILES,
        )
        self.rf.fit()

    def test_preprocess(self, frame):
        """
        Should produce expected columns, given raw data.
        """
        data = RFModel._preprocess(
            data=frame,
            months=self.rf.months,
            end_month_index=self.rf.end_month_index,
            date_column="time_end",
        )

        assert "season" in data.columns
        assert "geography" in data.columns
        assert data.shape[0] > 0  # should have at least 1 row

        time_cols = [col for col in data.columns if col not in ["season", "geography"]]

        assert len(time_cols) > 0  # at least 1 time column
        assert all(
            c.startswith("t=") for c in time_cols
        )  # all time columns should be named like "t=0", "t=1", etc.

    def test_fit(self):
        assert isinstance(self.rf.models, dict)
        assert all(
            isinstance(v, RandomForestRegressor) for v in self.rf.models.values()
        )

    def test_predict(self):
        pred = self.rf.predict()

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
