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
    p.add_argument("--forecasts", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    obs_raw = pl.read_parquet(args.data)
    forecasts_raw = pl.read_parquet(args.forecasts)
    scores = pl.read_parquet(args.scores)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    assert Path(args.output).name == "forecast_example.png"

    # get the forecast cone
    half_alpha = (1.0 - CI_LEVEL) / 2
    forecasts = forecasts_raw.group_by(
        ["season", "geography", "time_end", "model", "forecast_start"]
    ).agg(
        pl.col("estimate").median().alias("fc_estimate"),
        pl.col("estimate").quantile(half_alpha).alias("fc_lci"),
        pl.col("estimate").quantile(1.0 - half_alpha).alias("fc_uci"),
    )

    obs = obs_raw.select(
        ["season", "geography", "time_end", "estimate", "lci", "uci"]
    ).rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})

    # combine obs and fc's into "data", which is an abuse of terminology
    data = forecasts.join(obs, on=["season", "geography", "time_end"], how="left")

    # for one state, show forecasts (which are only one season)
    base = alt.Chart(data.filter(pl.col("geography") == pl.lit("New Jersey"))).encode(
        alt.X("time_end")
    )

    fc_cone = base.mark_area().encode(alt.Y("fc_lci"), alt.Y2("fc_uci"))
    fc_points = base.mark_line(color="red").encode(alt.Y("fc_estimate"))
    fc_data = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    fc_data_error = base.mark_rule(color="black").encode(
        alt.X2("time_end"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    (fc_cone + fc_points + fc_data + fc_data_error).facet(column="forecast_start").save(
        args.output
    )

    print(scores.filter(pl.col("forecast_start") == pl.col("forecast_start").max()))
