import argparse
from pathlib import Path
from typing import List

import altair as alt
import polars as pl
import yaml
from plot_data import (
    AXIS_PERCENT,
    MEDIAN_ENCODINGS,
    TICK_KWARGS,
    add_medians,
    gather_n,
    month_order,
)

LINE_OPACITY = 0.4


def plot_forecast(
    data: pl.DataFrame, geography: str, sort_month: List[str]
) -> alt.FacetChart:
    # remove forecast dates that have no actual forecasts
    # .filter(pl.col("forecast_date") < pl.col("time_end").max())

    # hack: at the last forecast date, we make a forecast for only one date, but we
    # can't show this with .mark_line(), and so it ends up blank. So remove those
    # dates
    good_fc_dates = (
        data.filter(pl.col("pred_estimate").is_not_null())
        .group_by("forecast_date")
        .agg(n=pl.col("time_end").unique().len())
        .filter(pl.col("n") >= 2)
        .select("forecast_date")
        .to_series()
        .to_list()
    )

    base = alt.Chart(
        data.filter(
            pl.col("geography") == pl.lit(geography),
            pl.col("forecast_date").is_in(good_fc_dates),
        )
    ).encode(alt.X("month", title=None, sort=sort_month))

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

    return (fc_cone + fc_points + fc_data + fc_data_error).facet(
        column=alt.Column("forecast_date", header=None)
    )


def plot_fit(
    data: pl.DataFrame,
    pred: pl.DataFrame,
    geography: str,
    season: str,
    sort_month: List[str],
) -> alt.LayerChart:
    assert pred["forecast_date"].unique().len() == 1
    base = alt.Chart(
        pred.filter(
            pl.col("geography") == pl.lit(geography), pl.col("season") == pl.lit(season)
        ).join(data, on=["season", "geography", "time_end"], how="inner")
    ).encode(alt.X("month", title=None, sort=sort_month))

    cone = base.mark_area(fill="black", opacity=0.25).encode(
        alt.Y("pred_lci", title="Coverage", axis=AXIS_PERCENT), alt.Y2("pred_uci")
    )
    fit_points = base.mark_line(color="black", opacity=0.75).encode(
        alt.Y("pred_estimate")
    )
    obs_points = base.mark_point(color="black").encode(alt.Y("obs_estimate"))
    obs_error = base.mark_rule(color="black").encode(
        alt.X2("month"), alt.Y("obs_lci"), alt.Y2("obs_uci")
    )

    return cone + fit_points + obs_points + obs_error


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
    preds = pl.scan_parquet(args.preds)
    scores = pl.read_parquet(args.scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # clean data
    obs = (
        obs_raw.select(["season", "geography", "time_end", "estimate", "lci", "uci"])
        .rename({"estimate": "obs_estimate", "lci": "obs_lci", "uci": "obs_uci"})
        .with_columns(month=pl.col("time_end").dt.to_string("%b"))
    )

    # get the prediction cones
    half_alpha = (1.0 - config["plots"]["ci_level"]) / 2
    pred_cones = (
        preds.group_by(["season", "geography", "time_end", "model", "forecast_date"])
        .agg(
            pl.col("estimate").median().alias("pred_estimate"),
            pl.col("estimate").quantile(half_alpha).alias("pred_lci"),
            pl.col("estimate").quantile(1.0 - half_alpha).alias("pred_uci"),
        )
        .collect()
    )

    # get all the target dates & forecast dates
    fc = pred_cones.filter(pl.col("time_end") > pl.col("forecast_date"))
    fc_plot_data = (
        obs.join(fc.select(pl.col("forecast_date").unique()), how="cross")
        .join(
            fc,
            on=["season", "geography", "time_end", "forecast_date"],
            how="left",
        )
        # keep only the last season
        .filter(pl.col("season") == pl.col("season").max())
    )

    sort_month = month_order(config["season"]["start_month"])
    enc_x_month = alt.X("month", title=None, sort=sort_month)

    # example fits and forecasts
    for geo in config["plots"]["example_geos"]:
        plot_fit(
            data=obs,
            pred=pred_cones.filter(
                pl.col("forecast_date") == pl.col("forecast_date").max()
            ),
            geography=geo,
            season=config["plots"]["example_season"],
            sort_month=sort_month,
        ).save(out_dir / f"fit_{geo}.svg")

        plot_forecast(data=fc_plot_data, geography=geo, sort_month=sort_month).save(
            out_dir / f"forecast_{geo}.svg"
        )

    # scores across seasons & states
    fit_scores = scores.filter(
        pl.col("forecast_date") == pl.col("forecast_date").max(),
        pl.col("score_fun") == pl.lit("mspe"),
    ).with_columns(pl.col("score_value").log())

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

    sis_line = (
        alt.Chart(sis_data)
        .mark_line(color="black", opacity=LINE_OPACITY)
        .encode(
            enc_x_month,
            alt.Y("score_value", title="Score (abs. end-of-season diff.)"),
            alt.Detail("geography"),
        )
    )

    sis_tick_base = alt.Chart(
        sis_data.filter(pl.col("forecast_date") == pl.col("forecast_date").max())
        .sort("score_value")
        .pipe(gather_n, 5)
    ).encode(enc_x_month, alt.Y("score_value"), alt.Text("geography"))

    sis_tick = sis_tick_base.mark_point(**TICK_KWARGS)
    sis_text = sis_tick_base.mark_text(align="left", dx=15)

    (sis_line + sis_tick + sis_text).save(out_dir / "scores_increasing.svg")

    ## summary of end-of-season abs diff ##
    alt.Chart(sis_data).mark_boxplot(extent="min-max").encode(
        enc_x_month,
        alt.Y("score_value", title="Score (abs. end-of-season diff.)"),
    ).save(out_dir / "eos_abs_diff_summary.svg")

    # end-of-season abs diff by state #
    state_sort = (
        sis_data.filter(pl.col("month") == "Jul")
        .sort(pl.col("score_value"))
        .select("geography")
        .to_numpy()
        .ravel()
        .tolist()
    )
    alt.Chart(sis_data).mark_line(color="black", opacity=LINE_OPACITY).encode(
        enc_x_month,
        alt.Y("score_value", title="Score (abs. end-of-season diff.)"),
        alt.Facet("geography", columns=9, sort=state_sort),
    ).save(out_dir / "eos_abs_diff_by_state.svg")

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
