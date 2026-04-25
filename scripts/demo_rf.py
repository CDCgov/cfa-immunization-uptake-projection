"""Random Forest demonstration script for immunization uptake projection."""

from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from plot_data import month_order
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from iup import date_to_season

# Create output directory
OUTPUT_DIR = Path("output/demo_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEASON_START_MONTH = 7  # July
END_MONTH = 9  # 9 months after July, i.e., April
MIN_T = 1 - END_MONTH
TARGET_SEASON = "2021/2022"
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
def forecast(forecast_t: int, target_season: str = TARGET_SEASON, data=data):
    assert forecast_t >= MIN_T

    enc = CoverageEncoder()
    enc.fit(data)

    # fit the model
    data_fit = data.filter(pl.col("season") != pl.lit(target_season))
    features = ["season", "geography"] + [
        f"t={t}" for t in range(MIN_T, forecast_t + 1)
    ]
    targets = [f"t={t}" for t in range(forecast_t + 1, 1)]

    X_fit = enc.encode(data_fit.select(features))

    y_fit = data_fit.select(targets).to_numpy()
    # special case: if predicting only the last data point
    if y_fit.shape[1] == 1:
        y_fit = y_fit.ravel()

    rf = RandomForestRegressor()
    rf.fit(X_fit, y_fit)

    # make the forecast
    data_pred = data.filter(pl.col("season") == pl.lit(target_season))
    X_pred = enc.encode(data_pred.select(features))
    y_pred = rf.predict(X_pred)
    if len(targets) == 1:
        y_pred = y_pred.reshape(-1, 1)

    return (
        data_pred.select(["season", "geography"])
        .with_columns(forecast_t=forecast_t)
        .hstack(pl.DataFrame(y_pred, schema=targets, orient="row"))
        .unpivot(
            on=targets,
            index=["season", "geography", "forecast_t"],
            variable_name="target_t",
            value_name="pred",
        )
        .with_columns(pl.col("target_t").str.replace("t=", "").cast(pl.Int64))
    )


# Generate forecasts
print("\nGenerating forecasts...")
forecasts = pl.concat(
    [forecast(x, target_season=TARGET_SEASON) for x in range(MIN_T, 0)]
)

truth = data.unpivot(
    on=[c for c in data.columns if c.startswith("t=")],
    index=["season", "geography"],
    variable_name="target_t",
    value_name="true",
).with_columns(target_t=pl.col("target_t").str.replace("t=", "").cast(pl.Int64))

target_truth = truth.filter(pl.col("season") == pl.lit(TARGET_SEASON))
plot_data = pl.concat(
    [
        forecasts.select(["season", "geography", "forecast_t", "target_t", "pred"])
        .rename({"pred": "estimate"})
        .with_columns(
            series=pl.format("forecast_t={}", pl.col("forecast_t").cast(pl.String))
        )
        .drop("forecast_t"),
        target_truth.select(["season", "geography", "target_t", "true"])
        .rename({"true": "estimate"})
        .with_columns(series=pl.lit("observed")),
    ],
    how="diagonal_relaxed",
)

# Forecast visualization by geography
chart3 = alt.layer(
    alt.Chart()
    .transform_filter(alt.datum.series != "observed")
    .mark_line(point=True, opacity=0.65)
    .encode(
        alt.X("target_t:Q", title="target_t"),
        alt.Y("estimate:Q", title="estimate"),
        alt.Color("series:N", title="series"),
        alt.Detail("series:N"),
    ),
    alt.Chart()
    .transform_filter(alt.datum.series == "observed")
    .mark_line(color="black", strokeWidth=2)
    .encode(
        alt.X("target_t:Q"),
        alt.Y("estimate:Q"),
    ),
    data=plot_data,
).facet(alt.Facet("geography:N"), columns=5)
chart3.save(str(OUTPUT_DIR / "forecasts_by_geography.png"))
print(f"Saved forecasts by geography to {OUTPUT_DIR / 'forecasts_by_geography.png'}")

# Forecast errors
errors = forecasts.join(truth, on=["season", "geography", "target_t"]).with_columns(
    error=pl.col("pred") - pl.col("true")
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
