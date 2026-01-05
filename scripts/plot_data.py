import argparse
import calendar
from pathlib import Path
from typing import List

import altair as alt
import polars as pl
import yaml

MEDIAN_POINT_KWARGS = {
    "color": "black",
    "shape": "stroke",
    "strokeWidth": 3,
    "size": 125,
}

MEDIAN_ENCODINGS = [
    alt.Color(
        "type",
        scale=alt.Scale(domain=["datum", "median"], range=["black", "red"]),
        legend=None,
    ),
    alt.Shape(
        "type",
        scale=alt.Scale(domain=["datum", "median"], range=["circle", "stroke"]),
    ),
    alt.Size("type", scale=alt.Scale(domain=["datum", "median"], range=[20, 200])),
]


def add_medians(
    df: pl.DataFrame,
    group_by: str,
    value_col: str = "estimate",
    type_col: str = "type",
) -> pl.DataFrame:
    return pl.concat(
        [
            df.with_columns(pl.lit("datum").alias(type_col)).select(
                [group_by, value_col, type_col]
            ),
            eos.group_by(group_by)
            .agg(pl.col(value_col).median())
            .with_columns(pl.lit("median").alias(type_col)),
        ]
    )


def month_order(season_start_month: int) -> List[str]:
    return [
        calendar.month_abbr[i]
        for i in list(range(season_start_month, 12 + 1))
        + list(range(1, season_start_month))
    ]


assert month_order(7) == [
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data = pl.read_parquet(args.data)

    # ensure output directory exists
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ensure the path is what we expect
    assert Path(args.output).name == "data_one_season_by_state.png"

    # for one season, show each state's trajectory
    alt.Chart(
        data.filter(pl.col("season") == pl.lit("2020/2021")).with_columns(
            month=pl.col("time_end").dt.to_string("%b")
        )
    ).mark_line().encode(
        alt.X("month", sort=month_order(config["season"]["start_month"])),
        alt.Y("estimate"),
        alt.Detail("geography"),
    ).mark_line().save(args.output)

    # end of season data
    eos = data.filter((pl.col("time_end") == pl.col("time_end").max()).over("season"))

    # for each season, show eos spread over states
    medians = (
        alt.Chart(eos.group_by("season").agg(pl.col("estimate").median()))
        .mark_point(**MEDIAN_POINT_KWARGS)
        .encode(alt.X("season"), alt.Y("estimate"))
    )

    points = alt.Chart(eos).mark_point().encode(alt.X("season"), alt.Y("estimate"))

    alt.Chart(add_medians(eos, "season")).mark_point().encode(
        alt.X("season"),
        alt.Y("estimate"),
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "data_eos_by_season.png")

    # for each state, show eos spread over seasons
    alt.Chart(add_medians(eos, "geography")).mark_point().encode(
        alt.X(
            "geography", sort=alt.EncodingSortField("estimate", "median", "descending")
        ),
        alt.Y("estimate"),
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "data_eos_by_state.png")
