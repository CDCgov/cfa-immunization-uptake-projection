"""Random Forest demonstration script for immunization uptake projection."""

from datetime import date
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

import iup.models
from iup import date_to_season


def to_inc(t: pl.Expr, estimate: pl.Expr) -> pl.Expr:
    return pl.struct(
        start=pl.lit(None).append(t.reverse().slice(1).reverse()),
        end=t,
        inc=estimate.diff(),
    )


# Create output directory
OUTPUT_DIR = Path("output/demo_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEASON_START_MONTH = 7  # July

# Load and prepare data
data = (
    pl.read_parquet("data/raw.parquet")
    .filter(pl.col("geography_type") == pl.lit("admin1"))
    .with_columns(
        season=date_to_season(
            pl.col("time_end"), season_start_month=SEASON_START_MONTH
        ),
        t=iup.models.LPLModel._days_in_season(
            pl.col("time_end"),
            season_start_month=SEASON_START_MONTH,
            season_start_day=1,
        ),
    )
    # remove partial seasons
    .filter(pl.col("season").is_in(["2008/2009", "2022/2023"]).not_())
    # convert to incident coverage
    .sort(["season", "geography"])
    .with_columns(inc=pl.col("estimate").diff())
    .select(["season", "geography", "time_end", "estimate", "start", "end", "inc"])
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


def forecast(forecast_date: date, data=data) -> pl.DataFrame:
    enc = CoverageEncoder()
    enc.fit(data)

    # fit the model
    data_fit = data.drop_nulls().filter(pl.col("time_end") <= forecast_date)
    X_fit = enc.encode(data_fit.select(["season", "geography", "start", "end"]))
    y_fit = data_fit["inc"]

    rf = RandomForestRegressor()
    rf.fit(X_fit, y_fit)

    # make the forecast
    data_pred = data.drop_nulls().filter(pl.col("time_end") > forecast_date)
    X_pred = enc.encode(data_pred.select(["season", "geography", "start", "end"]))
    y_pred = rf.predict(X_pred)

    # assemble the cumulative coverage based on incremental values
    data_start = (
        data.filter(pl.col("time_end") == forecast_date)
        .select("season", "geography", "estimate")
        .rename({"estimate": "start_estimate"})
    )

    preds = (
        data_pred.select(["season", "geography", "time_end"])
        .join(data_start, on=["season", "geography"], how="left")
        .with_columns(forecast_date=forecast_date, pred_inc=y_pred)
        .sort(["season", "geography", "time_end"])
        .with_columns(
            pred=pl.col("start_estimate")
            + pl.col("pred_inc").cum_sum().over(["season", "geography"])
        )
    )

    return preds


# Generate forecasts
print("\nGenerating forecasts...")
fc_dates = (
    data.filter(pl.col("season") == pl.col("season").max())
    .select(pl.col("time_end").unique())
    .sample(3)
    .to_series()
)
forecasts = pl.concat([forecast(x) for x in fc_dates])

print(forecasts)

# Forecast visualization by geography
chart_data = (
    data.filter(pl.col("season") == pl.col("season").max())
    .join(forecasts, on=["season", "geography", "time_end"], how="left")
    .with_columns(error=pl.col("pred") - pl.col("estimate"))
)
base = alt.Chart(chart_data)
points_data = base.mark_point().encode(alt.X("time_end"), alt.Y("estimate"))
fc = base.mark_line().encode(
    alt.X("time_end"), alt.Y("pred"), alt.Color("forecast_date", type="nominal")
)
chart3 = (fc + points_data).facet("geography", columns=5)
chart3.save(str(OUTPUT_DIR / "forecasts_by_geography.png"))
print(f"Saved forecasts by geography to {OUTPUT_DIR / 'forecasts_by_geography.png'}")

# Forecast errors
(
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        alt.X("error", bin=alt.Bin(step=0.01)),
        alt.Y("count()"),
        alt.Facet("forecast_date"),
    )
).save(str(OUTPUT_DIR / "forecast_errors.png"))
print(f"Saved forecast errors to {OUTPUT_DIR / 'forecast_errors.png'}")
