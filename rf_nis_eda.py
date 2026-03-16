from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from iup.utils import date_to_season
from scripts.plot_data import month_order

# Create output directory
OUTPUT_DIR = Path("output/demo_rf")
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
    alpha=0.05,
    n_trees: int = 100,
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

    # Fit training data #
    rf = RandomForestRegressor(n_estimators=n_trees)

    # make prediction from individual tree #
    data_pred = data.filter(pl.col("season") == pl.lit(target_season))
    X_pred = enc.encode(data_pred.select(features))

    preds = np.array(
        [tree.predict(X_pred) for tree in rf.fit(X_fit, y_fit).estimators_]
    )

    pred_mean = np.mean(preds, axis=0)
    pred_lpi = np.quantile(preds, alpha / 2, axis=0)
    pred_upi = np.quantile(preds, 1 - alpha / 2, axis=0)

    summary_pred = pl.DataFrame({"pred": pred_mean, "lpi": pred_lpi, "upi": pred_upi})

    all_pred = pl.concat(
        [data_pred.select(["season", "geography"]), summary_pred], how="horizontal"
    )

    all_pred = all_pred.with_columns(forecast_t=forecast_t, target_t=target_t)

    return all_pred


def plot(
    x_name,
    plot_name,
    data,
    obs_name="estimate:Q",
    pred_name="pred:Q",
    lpi_name="lpi:Q",
    upi_name="upi:Q",
    facet="geography:N",
):
    chart_data = alt.Chart(data).mark_point().encode(alt.X(x_name), alt.Y(obs_name))

    chart_pred = (
        alt.Chart(data)
        .mark_line(point=False, color="orange")
        .encode(
            alt.X(x_name),
            alt.Y(pred_name),
        )
    )

    chart_qs = (
        alt.Chart(data)
        .mark_area(opacity=0.3)
        .encode(alt.X(x_name), alt.Y(lpi_name), alt.Y2(upi_name))
    )
    charts = alt.layer(chart_data, chart_pred, chart_qs).facet(facet, columns=5)

    charts.save(OUTPUT_DIR / plot_name)


data_to_join = (
    data.unpivot(
        index=["season", "geography"], variable_name="forecast_t", value_name="estimate"
    )
    .rename({"forecast_t": "target_t"})
    .filter(pl.col("season") == pl.lit("2021/2022"))
)

### forecast end-of-season across forecast dates ###
forecasts = [forecast(x) for x in range(MIN_T, 1)]
forecasts = pl.concat(forecasts)

pred_data1 = (
    data_to_join.filter(pl.col("target_t") == pl.lit("t=0"))
    .sort(["season", "geography", "target_t"])
    .join(forecasts, on=["season", "geography"], how="right")
)

plot(x_name="forecast_t:Q", plot_name="rf_ci_vary_fct.png", data=pred_data1)


##### forecast a series ####
forecasts = [forecast(forecast_t=-8, target_t=x) for x in range(MIN_T, 1)]
forecasts = pl.concat(forecasts)

pred_data2 = (
    data_to_join.with_columns(
        target_t=pl.col("target_t").str.replace("t=", "").cast(pl.Int64)
    )
    .sort(["season", "geography", "target_t"])
    .join(forecasts, on=["season", "geography", "target_t"], how="right")
)

plot(x_name="target_t:Q", plot_name="rf_ci_fct=-8.png", data=pred_data2)
