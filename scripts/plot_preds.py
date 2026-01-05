import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml

CI_LEVEL = 0.95


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    obs_raw = pl.read_parquet(args.data)
    pred_samples = pl.read_parquet(args.preds)
    scores = pl.read_parquet(args.scores)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    assert Path(args.output).name == "forecast_example.png"

    # get the forecast cone
    half_alpha = (1.0 - CI_LEVEL) / 2
    preds = pred_samples.group_by(
        ["season", "geography", "time_end", "model", "forecast_start"]
    ).agg(
        pl.col("estimate").median().alias("pred_estimate"),
        pl.col("estimate").quantile(half_alpha).alias("pred_lci"),
        pl.col("estimate").quantile(1.0 - half_alpha).alias("pred_uci"),
    )

    obs = obs_raw.select(
        ["season", "geography", "time_end", "estimate", "lci", "uci"]
    ).rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})

    forecasts = preds.filter(pl.col("time_end") > pl.col("forecast_start"))

    # get all the target dates & forecast dates
    fc_plot = obs.join(
        forecasts.select(pl.col("forecast_start").unique()), how="cross"
    ).join(
        forecasts,
        on=["season", "geography", "time_end", "forecast_start"],
        how="left",
    )

    # for one state, show forecasts (which are only one season)
    base = alt.Chart(
        fc_plot.filter(
            pl.col("geography") == pl.lit("New Jersey"),
            pl.col("season") == pl.col("season").max(),
        )
    ).encode(alt.X("time_end"))

    fc_cone = base.mark_area().encode(alt.Y("pred_lci"), alt.Y2("pred_uci"))
    fc_points = base.mark_line(color="red").encode(alt.Y("pred_estimate"))
    fc_data = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    fc_data_error = base.mark_rule(color="black").encode(
        alt.X2("time_end"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    (fc_cone + fc_points + fc_data + fc_data_error).facet(column="forecast_start").save(
        args.output
    )

    # scores across seasons & states
    fit_scores = scores.filter(
        pl.col("forecast_start") == pl.col("forecast_start").max(),
        pl.col("score_type") == pl.lit("fit"),
        pl.col("score_fun") == pl.lit("mspe"),
    ).with_columns(pl.col("score_value").log())

    alt.Chart(fit_scores).mark_point().encode(
        alt.X("season"), alt.Y("score_value")
    ).save(out_dir / "score_by_season.png")

    alt.Chart(fit_scores).mark_point().encode(
        alt.X(
            "geography", sort=alt.EncodingSortField("estimate", "median", "descending")
        ),
        alt.Y("score_value"),
    ).save(out_dir / "score_by_geo.png")

    # scores increasing through the season?
    alt.Chart(
        scores.filter(
            pl.col("score_type") == pl.lit("forecast"),
            pl.col("score_fun") == pl.lit("eos_abs_diff"),
        )
    ).mark_line().encode(
        alt.X("forecast_start"), alt.Y("score_value"), alt.Color("geography")
    ).save(out_dir / "scores_increasing.png")

    # score vs. forecast
    avg_fit = (
        fit_scores.group_by(["model", "geography"])
        .agg(pl.col("score_value").median())
        .rename({"score_value": "fit_score"})
    )
    fc_goodness = (
        scores.filter(
            pl.col("score_type") == pl.lit("forecast"),
            pl.col("score_fun") == pl.lit("eos_abs_diff"),
            pl.col("forecast_start") == pl.col("forecast_start").min(),
        )
        .select(["geography", "model", "score_value"])
        .rename({"score_value": "fc_score"})
    )

    alt.Chart(
        avg_fit.join(fc_goodness, on=["model", "geography"], how="inner")
    ).mark_point().encode(alt.X("fit_score"), alt.Y("fc_score")).save(
        out_dir / "fc_fit_compare.png"
    )
