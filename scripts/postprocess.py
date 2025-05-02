import argparse
from typing import List

import altair as alt
import numpy as np
import polars as pl
import yaml

alt.data_transformers.disable_max_rows()


def plot_individual_projections(
    obs: pl.DataFrame, pred: pl.DataFrame, n_trajectories: int
):
    """
    Save a multiple-grid graph with the comparison between the observed uptake and the individual prediction projections,
    grouped by forecast start and model.

    Arguments:
    --------------
    obs: polars.Dataframe
        The observed uptake data frame, indicating the cumulative vaccine uptake as of `time_end`.
    pred: polars.Dataframe
        The predicted daily uptake, differed by forecast date, must include columns
        `forecast_start` and `estimate`.
    n_trajectories: int
        The number of prediction trajectories to plot.

    Return:
    -------------
        altair chart object
    """

    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    sample_ids = pred["sample_id"].unique()

    rng = np.random.default_rng(12345)

    selected_ids = rng.choice(sample_ids, size=n_trajectories)

    pred = pred.filter(pl.col("sample_id").is_in(selected_ids))

    # get every model/forecast date combo
    models_forecasts = pred.select(["model", "forecast_start"]).unique()

    # for every model and forecast date, merge in the observed value
    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique())
    )

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point(filled=True, color="black")
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month", title="Date"),
            ),
            alt.Y("estimate:Q"),
        )
    )

    pred_chart = (
        alt.Chart()
        .mark_line(opacity=0.3)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y(
                "mean(estimate):Q", axis=alt.Axis(title="Cumulative Uptake Estimate")
            ),
            color="sample_id:N",
        )
    )

    return alt.layer(pred_chart, obs_chart, data=pred).facet(
        column="model", row="forecast_start"
    )


def plot_summary(
    obs: pl.DataFrame, pred: pl.DataFrame, groups: List[str,], lci: float, uci: float
):
    """
    Save a multiple-grid graph of observed data and mean, interval estimate of prediction
    posterior distribution, grouped by model and forecast start.

    Arguments:
    --------------
    obs: polars.Dataframe
        The observed uptake data frame, indicating the cumulative vaccine uptake as of `time_end`.
    pred: polars.Dataframe
        The predicted daily uptake, differed by forecast date, must include columns
        `forecast_start` and `estimate`.
    config: yaml
        config file to specify the lower and upper bounds of credible interval to plot.

    Return:
    -------------
        altair chart object
    """

    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    models_forecasts = pred.select(["model", "forecast_start"]).unique()

    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique())
    )

    plot_pred = pred.with_columns(
        lower=pl.col("estimate")
        .quantile(lci / 100)
        .over(["model", "forecast_start", "time_end", "season", groups]),
        upper=pl.col("estimate")
        .quantile(uci / 100)
        .over(["model", "forecast_start", "time_end", "season", groups]),
    ).sort("time_end")

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point(color="black", filled=True)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("estimate:Q"),
        )
    )

    pred_chart = (
        alt.Chart()
        .mark_line()
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("mean(estimate):Q", title="Cumulative Uptake Estimate"),
        )
    )

    interval_chart = (
        alt.Chart()
        .mark_area(opacity=0.3)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            y="lower:Q",
            y2="upper:Q",
        )
    )

    return alt.layer(interval_chart, pred_chart, obs_chart, data=plot_pred).facet(
        column="model", row="forecast_start"
    )


def plot_score(scores: pl.DataFrame):
    """
    Save a evaluation score plot, varied by forecast start date.

    Arguments:
    --------------
    scores: polars.DataFrame
        The evaluation scores data frame.

    Return:
    -------------
        altair.chart object
    """
    score_names = scores["score_fun"].unique()

    score_dict = {
        "mspe": "Mean Squared Prediction Error",
        "mean_bias": "Mean Bias",
        "eos_abe": "End-of-season Absolute Error",
    }

    # every score name should have a label for the plot
    assert set(score_names).issubset(score_dict.keys())

    return (
        alt.Chart(scores.with_columns(pl.col("score_fun").replace_strict(score_dict)))
        .mark_point()
        .encode(
            alt.X("forecast_start:T", title="Forecast Start"),
            alt.Y("score:Q", title="Score"),
            alt.Column("score_fun", header=alt.Header(labelFontSize=20), title=None),
        )
        .resolve_scale(y="independent")
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--config", help="config file")
    p.add_argument("--eval", help="evaluation data")
    p.add_argument("--output", help="png file of forecast plots")
    p.add_argument("--scores", help="png file evaluation score plots")
    args = p.parse_args()

    pred = pl.read_parquet(args.pred)
    data = pl.read_parquet(args.obs)
    eval = pl.read_parquet(args.eval)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    plot_individual_projections(
        data, pred, config["forecast_plots"]["n_trajectories"]
    ).save(f"{args.output}trajectories.png")

    plot_summary(
        data,
        pred,
        config["data"]["groups"],
        config["forecast_plots"]["interval"]["lower"],
        config["forecast_plots"]["interval"]["upper"],
    ).save(f"{args.output}intervals.png")

    plot_score(eval).save(args.score)
