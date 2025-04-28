import argparse
from typing import List

import altair as alt
import numpy as np
import polars as pl
import yaml

alt.data_transformers.disable_max_rows()


def plot_individual_projections(
    obs: pl.DataFrame,
    pred: pl.DataFrame,
    n_trajectories: int,
    group_to_plot: List[str,],
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
    group_to_plot: List
        The grouping factors to plot. Must contain only 1 element.

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

    if group_to_plot is not None:
        assert len(group_to_plot) == 1
        ("Only one grouping factor is allowed for score plot.")
        group = group_to_plot[0]

    return alt.layer(pred_chart, obs_chart, data=pred).facet(
        row=alt.Row("forecast_start:T", title="Forecast start date"),
        column=alt.Column(group, type="nominal"),
    )


def plot_summary(
    obs: pl.DataFrame,
    pred: pl.DataFrame,
    groups: List[str,],
    lci: float,
    uci: float,
    group_to_plot: List[str,],
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
    groups: list
        A list of grouping factors.
    lci: float
        The quantile of the lower bound of the prediction interval. Must be between 0 and 1.
    uci: float
        The quantile of the upper bound of the prediction interval. Must be between 0 and 1.
    group_to_plot: list
        The list of grouping factors to plot. Must be only one element.

    Return:
    -------------
        altair chart object
    """

    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    assert len(pred["model"].unique()) == 1, "Only 1 model is allowed. "

    models_forecasts = pred.select(["model", "forecast_start"]).unique()

    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique())
    )
    
    plot_pred = pred.with_columns(
        lower=pl.col("estimate")
        .quantile(lci)
        .over(["model", "forecast_start", "time_end", groups]),
        upper=pl.col("estimate")
        .quantile(uci)
        .over(["model", "forecast_start", "time_end", groups]),
    ).sort("time_end")

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point(color="black", filled=True)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
                title="Date",
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

    if group_to_plot is not None:
        assert len(group_to_plot) == 1
        ("Only one grouping factor is allowed for score plot.")
        group = group_to_plot[0]

    return alt.layer(interval_chart, pred_chart, obs_chart, data=plot_pred).facet(
        column=alt.Column(group, type="nominal"),
        row=alt.Row("forecast_start:T", title="Forecast start date"),
    )


def plot_score(scores: pl.DataFrame, group_to_plot: List[str,]):
    """
    Save a evaluation score plot changed with forecast start date, grouped by model and score type.
    Only the metrics for forecast median is plotted.

    Arguments:
    --------------
    scores: polars.DataFrame
        The evaluation scores data frame.
    group_to_plot: list
        The list of grouping factors to plot. Must be only one element.

    Return:
    -------------
        altair.chart object
    """
    score_names = scores["score_name"].unique()

    score_dict = {
        "mspe": "Mean Squared Prediction Error",
    }

    for name in score_names:
        if name.startswith("abs_diff_"):
            score_dict[name] = "Absolute differenece at " + name[len("abs_diff_") :]

    # every score name should have a label for the plot
    assert set(score_names).issubset(score_dict.keys())

    plot_score = scores.filter(pl.col("quantile") == 0.5)

    if group_to_plot is not None:
        assert len(group_to_plot) == 1
        ("Only one grouping factor is allowed for score plot.")
        group = group_to_plot[0]

    return (
        alt.Chart(
            plot_score.with_columns(pl.col("score_name").replace_strict(score_dict))
        )
        .mark_point()
        .encode(
            alt.X("forecast_start:T", title="Forecast start date"),
            alt.Y("score_value:Q", title="Score value"),
            alt.Shape("quantile:N", title="Quantile"),
            alt.Row("score_name:N", header=alt.Header(labelFontSize=10), title=None),
            alt.Column(group, type="nominal", title=group),
        )
        .resolve_scale(y="independent")
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--score", help="evaluation scores")
    p.add_argument("--proj_output", help="png file of projection plot")
    p.add_argument("--score_output", help="png file of score plot")
    p.add_argument("--config", help="config file")
    p.add_argument("--summary_output", help="png file of mean and 95% CI plot")
    args = p.parse_args()

    pred = pl.read_parquet(args.pred)
    data = pl.read_parquet(args.obs)
    scores = pl.read_parquet(args.score)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pred = pred.filter(pl.col("model") == pl.lit("HillModel"))
    plot_individual_projections(
        data,
        pred,
        config["forecast_plots"]["n_trajectories"],
        config["scores"]["group_to_plot"],
    ).save(args.proj_output)

    plot_summary(
        data,
        pred,
        config["data"]["groups"],
        config["forecast_plots"]["interval"]["lower"],
        config["forecast_plots"]["interval"]["upper"],
        config["scores"]["group_to_plot"],
    ).save(args.summary_output)

    plot_score(scores, config["scores"]["group_to_plot"]).save(args.score_output)
