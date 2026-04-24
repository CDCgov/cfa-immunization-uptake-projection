import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml
from plot_data import (
    TICK_KWARGS,
    gather_n,
    month_order,
)

LINE_OPACITY = 0.4
NUMBER_HIGHLIGHT = 2
HIGHLIGHT_MONTH = "Jul"

# scores across seasons & states

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)

    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scores = pl.read_parquet(args.scores)

    out_flag = Path(args.output)
    out_dir = out_flag.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_scores = scores.filter(
        pl.col("forecast_date") == pl.col("forecast_date").max(),
        pl.col("score_fun") == pl.lit("mspe"),
    ).with_columns(pl.col("score_value").log())

    sort_month = month_order(config["season"]["start_month"])
    enc_x_month = alt.X("month:N", title=None, sort=sort_month)

    enc_y_mspe = alt.Y(
        "score_value", title="Score (Log(MSPE))", scale=alt.Scale(zero=False)
    )

    alt.Chart(fit_scores).mark_point().encode(
        alt.X("season", title=None),
        alt.Color("model"),
        enc_y_mspe,
    ).save(out_dir / "score_by_season.svg")

    alt.Chart(fit_scores).mark_point().encode(
        alt.X(
            "geography",
            title=None,
            sort=alt.EncodingSortField("score_value", "median", "descending"),
        ),
        enc_y_mspe,
        alt.Color("model"),
    ).save(out_dir / "score_by_geo.svg")

    # sis = score in season
    sis_data = scores.filter(
        pl.col("score_fun") == pl.lit("eos_abs_diff"),
        pl.col("season") == pl.col("season").max(),
    ).with_columns(month=pl.col("forecast_date").dt.to_string("%b"))

    line_encodings = [
        enc_x_month,
        alt.Y("score_value:Q", title="Score (abs. end-of-season diff.)"),
        alt.Detail("geography:Q"),
        alt.Color("model"),
    ]

    sis_line = (
        alt.Chart(sis_data)
        .mark_line(color="black", opacity=LINE_OPACITY)
        .encode(*line_encodings)
    )

    sis_tick_base = alt.Chart(
        sis_data.filter(pl.col("forecast_date") == pl.col("forecast_date").max())
        .sort("score_value")
        .pipe(gather_n, 3)
    ).encode(
        enc_x_month,
        alt.Y("score_value"),
        alt.Text("geography"),
    )

    sis_tick = sis_tick_base.mark_point(**TICK_KWARGS)
    sis_text = sis_tick_base.mark_text(align="left", dx=15)

    (sis_line + sis_tick + sis_text).save(out_dir / "scores_increasing.svg")

    out_flag.touch()
