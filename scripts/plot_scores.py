import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml

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

    # sis = score in season
    data = scores.filter(pl.col("score_fun") == pl.lit("eos_abs_diff"))

    base = alt.Chart(data).encode(
        alt.X("forecast_date", type="temporal", axis=alt.Axis(format="%b"))
    )
    line_chart = base.mark_line(point=True, opacity=LINE_OPACITY).encode(
        alt.Y("score_value", title="Score (abs. end-of-season diff.)"),
        alt.Detail("geography"),
        alt.Color("model"),
    )

    # tick_base = alt.Chart(
    #     sis_data.filter(pl.col("forecast_date") == pl.col("forecast_date").max())
    #     .sort("score_value")
    #     .pipe(gather_n, 3)
    # ).encode(
    #     enc_x_month,
    #     alt.Y("score_value"),
    #     alt.Text("geography"),
    # )

    # sis_tick = sis_tick_base.mark_point(**TICK_KWARGS)
    # sis_text = sis_tick_base.mark_text(align="left", dx=15)

    # (sis_line + sis_tick + sis_text).save(out_dir / "scores_increasing.svg")

    line_chart.save(out_dir / "scores.svg")

    out_flag.touch()
