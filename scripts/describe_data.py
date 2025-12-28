import argparse
from pathlib import Path

import altair as alt
import polars as pl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    data = pl.read_parquet(args.input)
    out_dir = Path(args.output_dir)

    # national, every month, every season
    alt.Chart(
        data.filter(pl.col("geography_type") == pl.lit("nation"))
    ).mark_line().encode(
        alt.X("t"),
        alt.Y("estimate"),
        alt.Detail("season"),
    ).mark_line().save(out_dir / "data_national.png")

    # state, every month, one season
    alt.Chart(
        data.filter(
            pl.col("geography_type") == pl.lit("admin1"),
            pl.col("season") == pl.lit("2015/2016"),
        )
    ).mark_line().encode(
        alt.X("t"),
        alt.Y("estimate"),
        alt.Detail("geography"),
    ).mark_line().save(out_dir / "data_state.png")

    # end of season
    eos = data.filter((pl.col("time_end") == pl.col("time_end").max()).over("season"))

    # # state & nation, last month, by season
    alt.Chart(eos).mark_point().encode(
        alt.X("season"), alt.Y("estimate"), alt.Color("geography_type")
    ).save(out_dir / "data_end_of_season.png")

    # # state & nation, last month, every season
    alt.Chart(eos).mark_point().encode(
        alt.X(
            "geography", sort=alt.EncodingSortField("estimate", "median", "descending")
        ),
        alt.Y("estimate"),
        alt.Color("geography_type"),
    ).save(out_dir / "data_end_state_season.png")
