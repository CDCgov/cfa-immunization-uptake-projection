import argparse
from pathlib import Path

import altair as alt
import polars as pl
import yaml
from plot_data import AXIS_PERCENT, MEDIAN_ENCODINGS, add_medians, month_order

ENC_Y_SCORE = alt.Y("score_value", title="Score (MSPE)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    obs_raw = pl.read_parquet(args.data)
    pred_samples = pl.read_parquet(args.preds)
    scores = pl.read_parquet(args.scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # get the forecast cone
    half_alpha = (1.0 - config["forecast_plots"]["ci_level"]) / 2
    preds = pred_samples.group_by(
        ["season", "geography", "time_end", "model", "forecast_date"]
    ).agg(
        pl.col("estimate").median().alias("pred_estimate"),
        pl.col("estimate").quantile(half_alpha).alias("pred_lci"),
        pl.col("estimate").quantile(1.0 - half_alpha).alias("pred_uci"),
    )

    obs = obs_raw.select(
        ["season", "geography", "time_end", "estimate", "lci", "uci"]
    ).rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})

    forecasts = preds.filter(pl.col("time_end") > pl.col("forecast_date"))

    # get all the target dates & forecast dates
    fc_plot = (
        obs.join(forecasts.select(pl.col("forecast_date").unique()), how="cross")
        .join(
            forecasts,
            on=["season", "geography", "time_end", "forecast_date"],
            how="left",
        )
        .with_columns(month=pl.col("time_end").dt.to_string("%b"))
    )

    enc_x_month = alt.X(
        "month", title=None, sort=month_order(config["season"]["start_month"])
    )

    # for one state, show forecasts (which are only one season)
    base = alt.Chart(
        fc_plot.filter(
            pl.col("geography") == pl.lit("New Jersey"),
            pl.col("season") == pl.col("season").max(),
        )
    ).encode(enc_x_month)

    fc_cone = base.mark_area(fill="black", opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT), alt.Y2("pred_uci")
    )
    fc_points = base.mark_line(color="black", opacity=0.75).encode(
        alt.Y("pred_estimate")
    )
    fc_data = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    fc_data_error = base.mark_rule(color="black").encode(
        alt.X2("month"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    (fc_cone + fc_points + fc_data + fc_data_error).facet(
        column=alt.Column("forecast_date", header=None)
    ).save(out_dir / "forecast_example.png")

    # scores across seasons & states
    fit_scores = scores.filter(
        pl.col("forecast_date") == pl.col("forecast_date").max(),
        pl.col("score_type") == pl.lit("fit"),
        pl.col("score_fun") == pl.lit("mspe"),
    ).with_columns(pl.col("score_value").log())

    alt.Chart(
        add_medians(fit_scores, group_by="season", value_col="score_value")
    ).mark_point().encode(
        alt.X("season", title=None), ENC_Y_SCORE, *MEDIAN_ENCODINGS
    ).save(out_dir / "score_by_season.png")

    alt.Chart(
        add_medians(fit_scores, group_by="geography", value_col="score_value")
    ).mark_point().encode(
        alt.X(
            "geography",
            title=None,
            sort=alt.EncodingSortField("estimate", "median", "descending"),
        ),
        ENC_Y_SCORE,
        *MEDIAN_ENCODINGS,
    ).save(out_dir / "score_by_geo.png")

    # scores increasing through the season?
    # sis = score in season
    sis_data = scores.filter(
        pl.col("score_type") == pl.lit("forecast"),
        pl.col("score_fun") == pl.lit("eos_abs_diff"),
    ).with_columns(month=pl.col("forecast_date").dt.to_string("%b"))

    sis_line = (
        alt.Chart(sis_data)
        .mark_line(
            color="black", opacity=0.5, point={"stroke": "black", "fill": "white"}
        )
        .encode(enc_x_month, ENC_Y_SCORE, alt.Detail("geography"))
    )

    sis_text = (
        alt.Chart(
            sis_data.filter(pl.col("forecast_date") == pl.col("forecast_date").max())
        )
        .mark_text(align="left", dx=15)
        .encode(enc_x_month, ENC_Y_SCORE, alt.Text("geography"))
    )

    (sis_line + sis_text).save(out_dir / "scores_increasing.png")

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
            pl.col("forecast_date") == pl.col("forecast_date").min(),
        )
        .select(["geography", "model", "score_value"])
        .rename({"score_value": "fc_score"})
    )

    alt.Chart(
        avg_fit.join(fc_goodness, on=["model", "geography"], how="inner")
    ).mark_point(color="black").encode(
        alt.X("fit_score", title="Fit score (MSPE)", scale=alt.Scale(zero=False)),
        alt.Y("fc_score", title="Forecast score (abs. end-of-season diff.)"),
    ).save(out_dir / "forecast_fit_compare.png")
