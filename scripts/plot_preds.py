import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml
from plot_data import AXIS_PERCENT, month_order

LINE_OPACITY = 0.4


def plot_forecast(
    obs: pl.DataFrame, pred_cones: pl.DataFrame, geography: str, sort_month: list[str]
) -> alt.FacetChart:
    fc = pred_cones.filter(
        pl.col("time_end") > pl.col("forecast_date"),
        pl.col("geography") == pl.lit(geography),
    )

    # hack: at the last forecast date, we make a forecast for only one date, but we
    # can't show this with .mark_line(), and so it ends up blank. So remove those
    # dates
    good_fc_dates = (
        fc.filter(pl.col("pred_estimate").is_not_null())
        .group_by("forecast_date")
        .agg(n=pl.col("time_end").unique().len())
        .filter(pl.col("n") >= 2)
        .select("forecast_date")
        .to_series()
        .to_list()
    )

    chart_data = (
        obs.filter(pl.col("geography") == pl.lit(geography))
        .join(pl.DataFrame({"forecast_date": good_fc_dates}), how="cross")
        .join(fc, on=["forecast_date", "time_end"], how="left")
    )

    base = alt.Chart(chart_data).encode(alt.X("month", title=None, sort=sort_month))
    fc_cone = base.mark_area(opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT),
        alt.Y2("pred_uci"),
        alt.Color("model"),
    )
    fc_points = base.mark_line(opacity=0.75).encode(
        alt.Y("pred_estimate"), alt.Color("model")
    )
    data_points = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    data_error = base.mark_rule(color="black").encode(
        alt.X2("month"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    return (fc_cone + fc_points + data_points + data_error).facet(
        column=alt.Column("forecast_date", header=None)
    )


def plot_fit(
    obs: pl.DataFrame,
    pred_cones: pl.DataFrame,
    geography: str,
    season: str,
    sort_month: list[str],
) -> alt.LayerChart:
    chart_data = pred_cones.filter(
        pl.col("forecast_date") == pl.col("forecast_date").max(),
        pl.col("geography") == pl.lit(geography),
        pl.col("season") == pl.lit(season),
    ).join(obs, on=["season", "geography", "time_end"], how="left")

    base = alt.Chart(chart_data).encode(alt.X("month", title=None, sort=sort_month))
    cone = base.mark_area(opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT),
        alt.Y2("pred_uci"),
        alt.Color("model"),
    )
    fit_points = base.mark_line(opacity=0.75).encode(
        alt.Y("pred_estimate"), alt.Color("model")
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

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = (
        pl.read_parquet(args.data)
        .filter(pl.col("season") == pl.col("season").max())
        .select(["season", "geography", "time_end", "estimate", "lci", "uci"])
        .rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})
        .with_columns(month=pl.col("time_end").dt.to_string("%b"))
    )
    preds = pl.read_parquet(args.preds).filter(
        pl.col("season") == pl.col("season").max()
    )

    # get the prediction cones
    half_alpha = config["alpha"] / 2
    quantiles = {half_alpha, 0.5, 1.0 - half_alpha}

    # check that quantiles are present in the data
    all_quantiles = preds.select(pl.col("quantile").unique()).to_series()
    assert quantiles.issubset(all_quantiles)

    pred_cones = (
        preds.filter(pl.col("quantile").is_in(quantiles))
        .with_columns(
            pl.col("quantile").replace_strict(
                {
                    half_alpha: "pred_lci",
                    0.5: "pred_estimate",
                    1.0 - half_alpha: "pred_uci",
                }
            )
        )
        .pivot(on="quantile", values="estimate")
    )

    sort_month = month_order(config["season"]["start_month"])

    # example fits and forecasts
    if "example_geos" in config["plots"] and config["plots"]["example_geos"]:
        geos = config["plots"]["example_geos"]
    else:
        geos = config["geographies"]

    for geo in geos:
        plot_fit(
            obs=obs,
            pred_cones=pred_cones,
            geography=geo,
            season=config["plots"]["example_season"],
            sort_month=sort_month,
        ).save(out_dir / f"fit_{geo}.svg")

        plot_forecast(
            obs=obs, pred_cones=pred_cones, geography=geo, sort_month=sort_month
        ).save(out_dir / f"forecast_{geo}.svg")

    out_flag.touch()
