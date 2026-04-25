"""Random Forest demonstration script for immunization uptake projection."""

import datetime
from pathlib import Path
from typing import Tuple

import altair as alt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from iup import date_to_season

SEASON_START_MONTH = 7  # July


def month_in_season(
    date: datetime.date,
    season_start_month: int = SEASON_START_MONTH,
    season_start_day: int = 1,
) -> int:
    assert date.day == 1
    year = date.year
    # start of a season that's in this year
    ssiy = datetime.date(year, season_start_month, season_start_day)

    # season start year
    if date < ssiy:
        ssy = year - 1
    else:
        ssy = year

    return (year - ssy) * 12 + (date.month - season_start_month)


# Create output directory
OUTPUT_DIR = Path("output/demo_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load and prepare data
data = (
    pl.read_parquet("data/raw.parquet")
    .filter(pl.col("geography_type") == pl.lit("admin1"))
    .with_columns(
        season=date_to_season(pl.col("time_end"), season_start_month=SEASON_START_MONTH)
    )
    # remove partial seasons
    .filter(pl.col("season").is_in(["2008/2009", "2022/2023"]).not_())
    .select(["season", "geography", "time_end", "estimate"])
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


# Forecasting function
def forecast(forecast_date: datetime.date, data=data):
    enc = CoverageEncoder()
    enc.fit(data)

    # add month in season
    data_t = data.with_columns(t=pl.col("time_end").map_elements(month_in_season))
    # keep track of season, month in season <-> date
    date_crosswalk = (
        data_t.select(["season", "time_end"])
        .unique()
        .with_columns(t=pl.col("time_end").map_elements(month_in_season))
    )
    # get the wide format data
    data_wide = (
        data_t.select(["season", "geography", "t", "estimate"])
        .pivot(on="t", values="estimate", sort_columns=True)
        # warning: this is a kludge
        .drop_nulls()
    )

    fc_season = pl.select(
        date_to_season(pl.lit(forecast_date), season_start_month=SEASON_START_MONTH)
    ).item()
    fc_month = month_in_season(forecast_date)

    X_features = ["season", "geography"] + [
        str(t) for t in range(0, fc_month + 1) if str(t) in data_wide.columns
    ]
    y_features = [
        str(t) for t in range(fc_month + 1, 12) if str(t) in data_wide.columns
    ]

    # fit the model
    data_fit = data_wide.filter(pl.col("season") <= fc_season)
    X_fit = enc.encode(data_fit.select(X_features))
    y_fit = data_fit.select(y_features).to_numpy()

    # sklearn complains if you pass a column vector rather than a 1d array
    if y_fit.shape[1] == 1:
        y_fit = y_fit.ravel()

    rf = RandomForestRegressor()
    rf.fit(X_fit, y_fit)

    # make the forecast
    data_pred = data_wide.filter(pl.col("season") >= fc_season)

    X_pred = enc.encode(data_pred.select(X_features))
    y_pred = rf.predict(X_pred)

    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    return (
        data_pred.select(["season", "geography"])
        .hstack(pl.DataFrame(y_pred, schema=y_features))
        .unpivot(
            on=y_features,
            index=["season", "geography"],
            variable_name="t",
            value_name="pred",
        )
        .with_columns(pl.col("t").cast(pl.Int64))
        .join(date_crosswalk, on=["season", "t"], how="left")
        .drop("t")
        .with_columns(forecast_date=forecast_date)
    )


# Generate forecasts
print("\nGenerating forecasts...")

fc_dates = (
    data.filter(pl.col("season") == pl.col("season").max())
    .select(pl.col("time_end").unique().sort())
    .to_series()
    # don't forecast at the last timepoint
    .head(-1)
)

forecasts = pl.concat([forecast(x) for x in fc_dates])

chart_data = (
    data.filter(pl.col("season") == pl.col("season").max())
    .join(pl.DataFrame({"forecast_date": fc_dates}), how="cross")
    .join(
        forecasts, on=["forecast_date", "season", "geography", "time_end"], how="left"
    )
    .with_columns(error=pl.col("pred") - pl.col("estimate"))
)

# Forecast visualization by geography
base = alt.Chart(chart_data).encode(alt.X("time_end"))
data_points = base.mark_point().encode(alt.Y("estimate"))
fc_line = base.mark_line().encode(alt.Y("pred"), alt.Color("forecast_date:O"))
(fc_line + data_points).facet("geography", columns=5).save(
    str(OUTPUT_DIR / "forecasts_by_geography.png")
)
print(f"Saved forecasts by geography to {OUTPUT_DIR / 'forecasts_by_geography.png'}")

# Forecast errors
errors = (
    data.filter(pl.col("season") == pl.col("season").max())
    .join(forecasts, on=["season", "geography", "time_end"], how="left")
    .with_columns(error=pl.col("pred") - pl.col("estimate"))
)

chart4 = (
    alt.Chart(errors)
    .mark_bar()
    .encode(
        alt.X("error", bin=alt.Bin(step=0.01)),
        alt.Y("count()"),
        alt.Facet("forecast_date"),
    )
)
chart4.save(str(OUTPUT_DIR / "forecast_errors.png"))
print(f"Saved forecast errors to {OUTPUT_DIR / 'forecast_errors.png'}")
