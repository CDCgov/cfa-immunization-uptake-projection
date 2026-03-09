import argparse
from pathlib import Path
from typing import List

import altair as alt
import polars as pl
import yaml
from plot_data import (
    AXIS_PERCENT,
    month_order,
)

alt.data_transformers.enable("vegafusion")

LINE_OPACITY = 0.4


def plot_forecast(
    data: pl.DataFrame, geography: str, sort_month: List[str]
) -> alt.FacetChart:
    # remove forecast dates that have no actual forecasts
    # .filter(pl.col("forecast_date") < pl.col("time_end").max())

    # hack: at the last forecast date, we make a forecast for only one date, but we
    # can't show this with .mark_line(), and so it ends up blank. So remove those
    # dates
    good_fc_dates = (
        data.filter(pl.col("pred_estimate").is_not_null())
        .group_by("forecast_date")
        .agg(n=pl.col("time_end").unique().len())
        .filter(pl.col("n") >= 2)
        .select("forecast_date")
        .to_series()
        .to_list()
    )

    base = alt.Chart(
        data.filter(
            pl.col("geography") == pl.lit(geography),
            pl.col("forecast_date").is_in(good_fc_dates),
        )
    ).encode(alt.X("month", title=None, sort=sort_month))

    fc_cone = base.mark_area(fill="black", opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT), alt.Y2("pred_uci")
    )
    fc_points = base.mark_line(color="black", opacity=0.75).encode(
        alt.Y("pred_estimate")
    )
    fc_data = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    fc_data_error = base.mark_rule(color="black").encode(
        alt.X2("month"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    return (fc_cone + fc_points + fc_data + fc_data_error).facet(
        column=alt.Column("forecast_date", header=None)
    )


def plot_fit(
    data: pl.DataFrame,
    pred: pl.LazyFrame,
    geography: str,
    season: str,
    sort_month: List[str],
) -> alt.LayerChart:
    pred_to_plot = (
        pred.filter(
            pl.col("forecast_date") == pl.col("forecast_date").max(),
            pl.col("geography") == pl.lit(geography),
            pl.col("season") == pl.lit(season),
        )
        .group_by(["season", "geography", "time_end", "model", "forecast_date"])
        .agg(
            pl.col("estimate").median().alias("pred_estimate"),
            pl.col("estimate").quantile(half_alpha).alias("pred_lci"),
            pl.col("estimate").quantile(1.0 - half_alpha).alias("pred_uci"),
        )
        .collect()
    )

    assert pred_to_plot["forecast_date"].unique().len() == 1

    base = alt.Chart(
        pred_to_plot.join(data, on=["season", "geography", "time_end"], how="inner")
    ).encode(alt.X("month", title=None, sort=sort_month))

    cone = base.mark_area(fill="black", opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT), alt.Y2("pred_uci")
    )
    fit_points = base.mark_line(color="black", opacity=0.75).encode(
        alt.Y("pred_estimate")
    )
    obs_points = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    obs_error = base.mark_rule(color="black").encode(
        alt.X2("month"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    return cone + fit_points + obs_points + obs_error


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--output", required=True, help="output flag file")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    obs_raw = pl.read_parquet(args.data)
    preds = pl.scan_parquet(args.preds)

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # clean data
    obs = (
        obs_raw.select(["season", "geography", "time_end", "estimate", "lci", "uci"])
        .rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})
        .with_columns(month=pl.col("time_end").dt.to_string("%b"))
    )

    # get the prediction cones
    half_alpha = (1.0 - config["plots"]["ci_level"]) / 2
    pred_cones = (
        preds.filter(pl.col("time_end") > pl.col("forecast_date"))
        .group_by(["season", "geography", "time_end", "model", "forecast_date"])
        .agg(
            pl.col("estimate").median().alias("pred_estimate"),
            pl.col("estimate").quantile(half_alpha).alias("pred_lci"),
            pl.col("estimate").quantile(1.0 - half_alpha).alias("pred_uci"),
        )
    )

    # get all the target dates & forecast dates
    fc = pred_cones.collect()

    fc_plot_data = (
        obs.join(fc.select(pl.col("forecast_date").unique()), how="cross")
        .join(
            fc,
            on=["season", "geography", "time_end", "forecast_date"],
            how="left",
        )
        # keep only the last season
        .filter(pl.col("season") == pl.col("season").max())
    )

    sort_month = month_order(config["season"]["start_month"])
    enc_x_month = alt.X("month", title=None, sort=sort_month)

    # example fits and forecasts
    for geo in config["plots"]["example_geos"]:
        plot_fit(
            data=obs,
            pred=preds,
            geography=geo,
            season=config["plots"]["example_season"],
            sort_month=sort_month,
        ).save(out_dir / f"fit_{geo}.svg")

        plot_forecast(data=fc_plot_data, geography=geo, sort_month=sort_month).save(
            out_dir / f"forecast_{geo}.svg"
        )

    out_flag.touch()
