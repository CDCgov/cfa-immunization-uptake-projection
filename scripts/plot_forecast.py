import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml


def get_cone(df: pl.DataFrame, level: float = 0.95):
    half_alpha = (1.0 - level) / 2
    df.group_by(["time_end", "model", "forecast_date"]).agg(
        pl.col("estimate").median().alias("median"),
        pl.col("estimate").quantile(half_alpha).alias("lci"),
        pl.col("estimate").quantile(1.0 - half_alpha).alias("uci"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--forecasts", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data = pl.read_parquet(args.data)
    forecasts = pl.read_parquet(args.forecasts)
    out_dir = Path(args.output_dir)

    # show one example forecast
    print(data)
    print(forecasts)
    raise RuntimeError

    # for one season, show each state's trajectory
    alt.Chart(data.filter(pl.col("season") == pl.lit("2015/2016"))).mark_line().encode(
        alt.X("t"),
        alt.Y("estimate"),
        alt.Detail("geography"),
    ).mark_line().save(out_dir / "data_one_season_by_state.png")

    # end of season data
    eos = data.filter((pl.col("time_end") == pl.col("time_end").max()).over("season"))

    # for each season, show eos spread over states
    alt.Chart(eos).mark_point().encode(alt.X("season"), alt.Y("estimate")).save(
        out_dir / "data_eos_by_state.png"
    )

    # for each state, show eos spread over seasons
    alt.Chart(eos).mark_point().encode(
        alt.X(
            "geography", sort=alt.EncodingSortField("estimate", "median", "descending")
        ),
        alt.Y("estimate"),
        alt.Color("geography_type"),
    ).save(out_dir / "data_eos_by_season.png")
