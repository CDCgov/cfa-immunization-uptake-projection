"""Random Forest demonstration script for immunization uptake projection."""

from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from plot_data import month_order
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from iup.utils import date_to_season

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

print("Data shape:", data.shape)
print("\nData description:")
print(data.describe())


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


# Train initial model
rf = RandomForestRegressor(oob_score=True)
enc = CoverageEncoder()

fit_data = data.drop("t=0")
enc.fit(fit_data)

X = enc.encode(fit_data)
y = data.select("t=0").to_series().to_numpy()

rf.fit(X, y)

# Feature importance analysis
features = pl.from_records(
    enc.categories(fit_data), orient="row", schema=["feature", "value"]
).with_columns(importance=rf.feature_importances_)

print("\nFeature importances:")
print(features)
print("\nFeature importance by group:")
print(features.group_by("feature").agg(pl.col("importance").sum()))

# Post hoc (but OOB) comparison of end-of-season predictions
chart1 = (
    alt.Chart(data.with_columns(pred=rf.oob_prediction_))
    .mark_point()
    .encode(alt.X("t=0"), alt.Y("pred"))
)
chart1.save(str(OUTPUT_DIR / "oob_predictions.png"))
print(f"\nSaved OOB predictions chart to {OUTPUT_DIR / 'oob_predictions.png'}")

# Distribution of end-of-season errors
chart2 = (
    alt.Chart(
        data.with_columns(pred=rf.oob_prediction_).with_columns(
            error=pl.col("pred") - pl.col("t=0")
        )
    )
    .mark_bar()
    .encode(alt.X("error", bin=True), alt.Y("count()"))
)
chart2.save(str(OUTPUT_DIR / "error_distribution.png"))
print(f"Saved error distribution chart to {OUTPUT_DIR / 'error_distribution.png'}")


# Forecasting function
def forecast(
    forecast_t: int, target_t: int = 0, target_season: str = "2021/2022", data=data
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

    rf = RandomForestRegressor()
    rf.fit(X_fit, y_fit)

    # make the forecast
    data_pred = data.filter(pl.col("season") == pl.lit(target_season))
    X_pred = enc.encode(data_pred.select(features))
    y_pred = rf.predict(X_pred)

    preds = data_pred.select(["season", "geography"]).with_columns(
        forecast_t=forecast_t, pred=y_pred
    )

    features = pl.from_records(
        enc.categories(data_fit.select(features)),
        orient="row",
        schema=["feature", "value"],
    ).with_columns(forecast_t=forecast_t, importance=rf.feature_importances_)

    return preds, features


# Generate forecasts
print("\nGenerating forecasts...")
results = [forecast(x) for x in range(MIN_T, 0 + 1)]
forecasts = pl.concat([x[0] for x in results])
importances = pl.concat([x[1] for x in results])

# Forecast visualization by geography
chart3 = (
    alt.Chart(forecasts)
    .mark_line(point=True)
    .encode(alt.X("forecast_t"), alt.Y("pred"), alt.Facet("geography", columns=5))
)
chart3.save(str(OUTPUT_DIR / "forecasts_by_geography.png"))
print(f"Saved forecasts by geography to {OUTPUT_DIR / 'forecasts_by_geography.png'}")

# Forecast errors
errors = (
    forecasts.filter(pl.col("forecast_t") != 0)
    .join(
        forecasts.filter(pl.col("forecast_t") == 0)
        .drop("forecast_t")
        .rename({"pred": "true"}),
        on=["season", "geography"],
    )
    .with_columns(error=pl.col("pred") - pl.col("true"))
)

chart4 = (
    alt.Chart(errors)
    .mark_bar()
    .encode(
        alt.X("error", bin=alt.Bin(step=0.01)),
        alt.Y("count()"),
        alt.Facet("forecast_t"),
    )
)
chart4.save(str(OUTPUT_DIR / "forecast_errors.png"))
print(f"Saved forecast errors to {OUTPUT_DIR / 'forecast_errors.png'}")

# Feature importance over time
chart5 = (
    alt.Chart(
        importances.group_by(
            pl.col("forecast_t"),
            pl.col("feature").replace({"unencoded": "in-season coverage"}),
        ).agg(pl.col("importance").sum())
    )
    .mark_line()
    .encode(alt.X("forecast_t"), alt.Y("importance"), alt.Color("feature"))
)
chart5.save(str(OUTPUT_DIR / "feature_importance_over_time.png"))
print(
    f"Saved feature importance over time to {OUTPUT_DIR / 'feature_importance_over_time.png'}"
)
