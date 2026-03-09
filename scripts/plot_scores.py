import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml
from plot_data import (
    MEDIAN_ENCODINGS,
    TICK_KWARGS,
    add_medians,
    gather_n,
    hightlight_state,
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

    alt.Chart(
        add_medians(fit_scores, group_by="season", value_col="score_value")
    ).mark_point().encode(
        alt.X("season", title=None),
        enc_y_mspe,
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "score_by_season.svg")

    alt.Chart(
        add_medians(fit_scores, group_by="geography", value_col="score_value")
    ).mark_point().encode(
        alt.X(
            "geography",
            title=None,
            sort=alt.EncodingSortField("score_value", "median", "descending"),
        ),
        enc_y_mspe,
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "score_by_geo.svg")

    # scores increasing through the season?
    # sis = score in season
    sis_data = scores.filter(
        pl.col("score_fun") == pl.lit("eos_abs_diff"),
        pl.col("season") == pl.col("season").max(),
    ).with_columns(month=pl.col("forecast_date").dt.to_string("%b"))

    line_encodings = dict(
        x=enc_x_month,
        y=alt.Y("score_value:Q", title="Score (abs. end-of-season diff.)"),
        detail=alt.Detail("geography:Q"),
    )

    sis_line = (
        alt.Chart(sis_data)
        .mark_line(color="black", opacity=LINE_OPACITY)
        .encode(**line_encodings)
    )

    sis_tick_base = alt.Chart(
        sis_data.filter(pl.col("forecast_date") == pl.col("forecast_date").max())
        .sort("score_value")
        .pipe(gather_n, 5)
    ).encode(
        enc_x_month,
        alt.Y("score_value"),
        alt.Text("geography"),
    )

    sis_tick = sis_tick_base.mark_point(**TICK_KWARGS)
    sis_text = sis_tick_base.mark_text(align="left", dx=15)

    (sis_line + sis_tick + sis_text).save(out_dir / "scores_increasing.svg")

    ## highlight the NUMBER_HIGHLIGHT of states that has lowest(best) and highest(worst) score in Jul
    sis_highlight_best = (
        alt.Chart(
            sis_data.filter(
                pl.col("geography").is_in(
                    hightlight_state(sis_data, HIGHLIGHT_MONTH, NUMBER_HIGHLIGHT)[
                        "best"
                    ]
                )
            )
        )
        .mark_line(color="blue")
        .encode(**line_encodings)
    )

    sis_highlight_worst = (
        alt.Chart(
            sis_data.filter(
                pl.col("geography").is_in(
                    hightlight_state(sis_data, HIGHLIGHT_MONTH, NUMBER_HIGHLIGHT)[
                        "worst"
                    ]
                )
            )
        )
        .mark_line(color="red")
        .encode(**line_encodings)
    )

    sis_tick_base_highlight = alt.Chart(
        sis_data.filter(
            pl.col("geography").is_in(
                hightlight_state(sis_data, HIGHLIGHT_MONTH, NUMBER_HIGHLIGHT)["best"]
                + hightlight_state(sis_data, HIGHLIGHT_MONTH, NUMBER_HIGHLIGHT)["worst"]
            ),
            pl.col("forecast_date") == pl.col("forecast_date").max(),
        )
    ).encode(enc_x_month, alt.Y("score_value"), alt.Text("geography"))

    sis_tick_highlight = sis_tick_base_highlight.mark_point(**TICK_KWARGS)
    sis_text_highlight = sis_tick_base_highlight.mark_text(align="left", dx=15)

    (
        sis_line
        + sis_highlight_best
        + sis_highlight_worst
        + sis_tick_highlight
        + sis_text_highlight
    ).save(out_dir / "scores_increasing_highlight.svg")

    # score vs. forecast
    avg_fit = (
        fit_scores.group_by(["model", "geography"])
        .agg(pl.col("score_value").median())
        .rename({"score_value": "fit_score"})
    )
    fc_goodness = (
        scores.filter(
            pl.col("score_fun") == pl.lit("eos_abs_diff"),
            pl.col("season") == pl.col("season").max(),
            pl.col("forecast_date") == pl.col("forecast_date").min(),
        )
        .select(["geography", "model", "score_value"])
        .rename({"score_value": "fc_score"})
    )

    alt.Chart(
        avg_fit.join(fc_goodness, on=["model", "geography"], how="inner")
    ).mark_point(color="black").encode(
        alt.X(
            "fit_score",
            title="Fit score (median MSPE over seasons)",
            scale=alt.Scale(zero=False),
        ),
        alt.Y("fc_score", title="Forecast score (abs. end-of-season diff.)"),
    ).save(out_dir / "forecast_fit_compare.svg")

    out_flag.touch()
