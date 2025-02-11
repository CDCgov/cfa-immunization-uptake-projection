import argparse

import altair as alt
import polars as pl
import yaml


def plot_projections(obs, pred, n_columns):
    """
    Save a multiple-grid graph with the comparison between the observed uptake and the prediction,
    initiated over the season.

    Arguments:
    --------------
    obs: polars.Dataframe
        The observed uptake data frame, indicating the cumulative vaccine uptake as of `time_end`.
    pred: polars.Dataframe
        The predicted daily uptake, differed by forecast date, must include columns `forecast_start` and `estimate`.
    n_columns: int
        The number of columns in the multiple-grid graph.

    Return:
    -------------
    None. The graph is saved.

    """

    # input check #
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    # plot weekly initiated prediction #
    time_axis = alt.Axis(format="%Y-%m", tickCount="month")

    obs_chart = (
        alt.Chart(obs)
        .mark_circle(color="black")
        .encode(alt.X("time_end", axis=time_axis), alt.Y("estimate"))
    )

    pred = pred.with_columns(
        date_str=("Forecast Date:" + pl.col("forecast_start").cast(pl.Utf8))
    )

    pred_chart = (
        alt.Chart()
        .mark_line(color="red")
        .encode(
            x="time_end:T",
            y="estimate:Q",
        )
    )

    return (
        alt.layer(obs_chart, pred_chart, data=pred)
        .facet(
            facet=alt.Facet(
                "date_str:N", title=None, header=alt.Header(labelFontSize=16)
            ),
            columns=n_columns,
        )
        .resolve_axis(x="independent")
    )


def plot_score(scores):
    """
    Save a evaluation score plot, varied by forecast start date.

    Arguments:
    --------------
    scores: polars.DataFrame
        The evaluation scores data frame.
    config: dict
        config.yaml to decide if plot or not.

    Return:
    -------------
    None. The graph is saved.

    """
    score_names = scores["score_fun"].unique()

    score_dict = {
        "mspe": "Mean Squared Prediction Error",
        "mean_bias": "Mean Bias",
        "eos_abe": "End-of-season Absolute Error",
    }

    charts = []
    for score_name in score_names:
        score = scores.filter(pl.col("score_fun") == score_name)

        score_chart = (
            alt.Chart(score)
            .mark_point()
            .encode(
                alt.X("forecast_start:T", title="Forecast Start"),
                alt.Y("score:Q", title="Score"),
            )
            .properties(title=score_dict[score_name])
        )

        charts = charts + [score_chart]

    return alt.hconcat(*charts).configure_title(fontSize=20)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--score", help="evaluation scores")
    p.add_argument("--proj_output", help="png file of projection plot")
    p.add_argument("--score_output", help="png file of score plot")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pred = pl.scan_parquet(args.pred).collect()
    data = pl.scan_parquet(args.obs).collect()

    if config["projection_plot"]["plot"]:
        plot_projections(data, pred, 4).save(args.proj_output)

    score = pl.scan_parquet(args.score).collect()

    if config["score_plot"]["plot"]:
        plot_score(score).save(args.score_output)
