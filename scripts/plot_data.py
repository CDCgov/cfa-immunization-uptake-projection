import argparse
import calendar
from pathlib import Path
from typing import List

import altair as alt
import numpy as np
import polars as pl
import yaml

AXIS_PERCENT = alt.Axis(format=".0%")
LINE_OPACITY = 0.25

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

TICK_KWARGS = {
    "shape": "stroke",
    "size": 50,
    "color": "black",
    "strokeWidth": 2,
    "opacity": 1,
}


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
            df.group_by(group_by)
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


def gather_n(df: pl.DataFrame, n: int, col_name="_idx") -> pl.DataFrame:
    """Take `n` evenly spaced rows from `df`, including the first and last"""
    assert n > 2
    assert df.height >= n, f"Asked for {n} rows, only have {df.height}"
    return (
        df.with_row_index(col_name)
        .filter(pl.col(col_name).is_in(np.linspace(0, df.height - 1, num=n).round()))
        .drop(col_name)
    )


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
    out_flag = Path(args.output)
    out_dir = out_flag.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # for one season, show each state's trajectory
    # exs = example season
    exs_data = data.filter(
        pl.col("season") == pl.lit(config["plots"]["example_season"])
    ).with_columns(month=pl.col("time_end").dt.to_string("%b"))

    enc_x_exs = alt.X(
        "month", title=None, sort=month_order(config["season"]["start_month"])
    )

    exs_line = (
        alt.Chart(exs_data)
        .mark_line(color="black", opacity=LINE_OPACITY)
        .encode(
            enc_x_exs,
            alt.Y("estimate", title="Coverage", axis=AXIS_PERCENT),
            alt.Detail("geography"),
        )
    )

    exs_tick_base = alt.Chart(
        exs_data.filter(pl.col("time_end") == pl.col("time_end").max())
        .sort("estimate")
        .pipe(gather_n, 5)
    ).encode(enc_x_exs, alt.Y("estimate"), alt.Text("geography"))

    exs_text = exs_tick_base.mark_text(align="left", dx=15)
    exs_tick = exs_tick_base.mark_point(**TICK_KWARGS)

    (exs_line + exs_tick + exs_text).save(out_dir / "coverage_trajectories.svg")

    # end of season data
    eos = data.filter((pl.col("time_end") == pl.col("time_end").max()).over("season"))

    # for each season, show eos spread over states
    alt.Chart(add_medians(eos, "season")).mark_point().encode(
        alt.X("season", title=None),
        alt.Y("estimate", title="End of season coverage", axis=AXIS_PERCENT),
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "coverage_by_season.svg")

    # for each state, show eos spread over seasons
    alt.Chart(add_medians(eos, "geography")).mark_point().encode(
        alt.X(
            "geography",
            title=None,
            sort=alt.EncodingSortField("estimate", "median", "descending"),
        ),
        alt.Y("estimate", title="End of season coverage", axis=AXIS_PERCENT),
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "coverage_by_state.svg")

    out_flag.touch()
