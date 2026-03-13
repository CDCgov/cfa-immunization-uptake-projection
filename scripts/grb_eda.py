from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

from iup.utils import date_to_season
from scripts.plot_data import month_order

# Create output directory
OUTPUT_DIR = Path("output/demo_grb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEASON_START_MONTH = 7  # July
END_MONTH = 9  # 9 months after July, i.e., April
MIN_T = 1 - END_MONTH
months = month_order(season_start_month=SEASON_START_MONTH)

# Load and prepare data
data = (
    pl.read_parquet("data/raw.parquet")
    .filter(pl.col("geography_type") == pl.lit("admin1"))
    .with_columns(
        season=date_to_season(
            pl.col("time_end"), season_start_month=SEASON_START_MONTH
        ),
        t=pl.col("time_end")
        .dt.to_string("%b")
        .map_elements(lambda x: months.index(x) - END_MONTH, pl.Int64),
    )
    .filter(pl.col("t").is_between(MIN_T, 0))
    # remove partial seasons
    .filter(pl.col("season").is_in(["2008/2009", "2022/2023"]).not_())
    .select(["season", "geography", "t", "estimate"])
    # go to long format
    .with_columns(pl.format("t={}", pl.col("t")))
    .pivot(on="t", values="estimate")
    # this is a kludge: really should impute these values
    .drop_nulls()
    .sort(["season", "geography"])
)


class CoverageEncoder:
    def __init__(self, categorical_feature_names: Tuple = ("season", "geography")):
        self.categorical_feature_names = categorical_feature_names
        self.enc = OneHotEncoder(sparse_output=False)
        self.categorical_features = None

    def fit(self, data: pl.DataFrame):
        self.enc.fit(data.select(self.categorical_feature_names).to_numpy())

        self.categorical_features = list(
            self._iter_features(self.categorical_feature_names, self.enc.categories_)
        )

    @staticmethod
    def _iter_features(names, categories):
        for feature, values in zip(names, categories):
            for value in values:
                yield (feature, value)

    def encode(self, data: pl.DataFrame) -> np.ndarray:
        X_enc = self.enc.transform(
            data.select(self.categorical_feature_names).to_numpy()
        )
        X_pass = data.drop(self.categorical_feature_names).to_numpy()

        assert isinstance(X_enc, np.ndarray)
        return np.asarray(np.hstack((X_enc, X_pass)))

    def categories(self, data: pl.DataFrame):
        if self.categorical_features is None:
            raise RuntimeError
        else:
            return self.categorical_features + [
                ("unencoded", col)
                for col in data.drop(self.categorical_feature_names).columns
            ]


# Forecasting function
def forecast(
    forecast_t: int,
    target_t: int = 0,
    target_season: str = "2021/2022",
    data=data,
    qs=[0.025, 0.5, 0.975],
):
    assert forecast_t >= MIN_T

    enc = CoverageEncoder()
    enc.fit(data)

    # fit the model
    data_fit = data.filter(pl.col("season") != pl.lit(target_season))
    features = ["season", "geography"] + [
        f"t={t}" for t in range(MIN_T, forecast_t + 1)
    ]

    X_fit = enc.encode(data_fit.select(features))
    y_fit = data_fit.select(f"t={target_t}").to_series().to_numpy()

    # Regression #
    grb_se = GradientBoostingRegressor(loss="squared_error")
    grb_se.fit(X_fit, y_fit)

    # make the forecast
    data_pred = data.filter(pl.col("season") == pl.lit(target_season))
    X_pred = enc.encode(data_pred.select(features))
    y_pred_se = grb_se.predict(X_pred)

    # Quantile regression #
    pred_dic = {}

    for q in qs:
        gbr_q = GradientBoostingRegressor(loss="quantile", alpha=q).fit(X_fit, y_fit)
        pred = gbr_q.predict(X_pred)
        pred_dic[f"quantile={q}"] = pred

    pred_qs = pl.DataFrame(pred_dic)

    preds = pl.concat(
        [data_pred.select(["season", "geography"]), pred_qs], how="horizontal"
    )

    preds = preds.with_columns(forecast_t=forecast_t, pred=y_pred_se)

    return preds


forecasts = [forecast(x) for x in range(MIN_T, 1)]
forecasts = pl.concat(forecasts)

data = (
    data.unpivot(
        index=["season", "geography"], variable_name="forecast_t", value_name="estimate"
    )
    .rename({"forecast_t": "target_t"})
    .filter(
        pl.col("season") == pl.lit("2021/2022"), pl.col("target_t") == pl.lit("t=0")
    )
    .with_columns(forecast_t=pl.col("target_t").str.replace("t=", "").cast(pl.Int64))
    .drop("forecast_t")
    .sort(["season", "geography", "target_t"])
)

pred_data = data.join(forecasts, on=["season", "geography"], how="right").rename(
    {"quantile=0.025": "lpi", "quantile=0.5": "median", "quantile=0.975": "upi"}
)

chart_data = (
    alt.Chart(pred_data).mark_point().encode(alt.X("forecast_t"), alt.Y("estimate"))
)

chart_pred = (
    alt.Chart(pred_data)
    .mark_line(point=False, color="orange")
    .encode(
        alt.X("forecast_t"),
        alt.Y("pred"),
    )
)

chart_med = (
    alt.Chart(pred_data)
    .mark_line(point=False, color="red")
    .encode(
        alt.X("forecast_t"),
        alt.Y("median"),
    )
)


chart_qs = (
    alt.Chart(pred_data)
    .mark_area(opacity=0.3)
    .encode(alt.X("forecast_t"), alt.Y("lpi"), alt.Y2("upi"))
)
charts = alt.layer(chart_data, chart_pred, chart_med, chart_qs).facet(
    "geography", columns=5
)

charts.save(str(OUTPUT_DIR / "grb_eda.png"))
