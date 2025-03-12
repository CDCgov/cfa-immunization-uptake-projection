import argparse

import altair as alt
import polars as pl
from altair import datum

alt.data_transformers.disable_max_rows()


def plot_individual_projections(obs: pl.DataFrame, pred: pl.DataFrame):
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
    config: yaml
        config file to specify grouping factors.

    Return:
    -------------
    chart object
    """

    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    # get every model/forecast date combo
    models_forecasts = pred.select(["model", "forecast_start", "sample_id"]).unique()

    # for every model and forecast date, merge in the observed value
    # column "type" will be either "obs" or "pred"
    plot_obs = (
        obs.join(models_forecasts, how="cross")
        .with_columns(type=pl.lit("obs"))
        .filter(pl.col("season").is_in(pred["season"].unique()))
    )

    plot_pred = pred.with_columns(pl.lit("pred").alias("type")).select(plot_obs.columns)

    plot_data = pl.concat([plot_obs, plot_pred])

    plot_new_data = plot_data.filter(
        pl.col("time_end").is_between(
            pl.col("forecast_start").min(),
            pl.col("forecast_start").max(),
            "both",
        )
    )

    obs_chart = (
        alt.Chart(plot_new_data)
        .mark_point(filled=True, color="black")
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month", title="Date"),
            ),
            alt.Y("estimate:Q"),
        )
        .transform_filter(datum.type == "obs")
    )

    pred_chart = (
        alt.Chart()
        .mark_line()
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("estimate:Q", axis=alt.Axis(title="Estimate")),
            color="sample_id:N",
        )
        .transform_filter(datum.type == "pred")
    )

    return alt.layer(pred_chart, obs_chart, data=plot_new_data).facet(
        column="model", row="forecast_start"
    )


def plot_summary(obs, pred):
    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    models_forecasts = pred.select(["model", "forecast_start", "sample_id"]).unique()

    plot_obs = (
        obs.join(models_forecasts, how="cross")
        .with_columns(type=pl.lit("obs"))
        .filter(pl.col("season").is_in(pred["season"].unique()))
    )

    plot_pred = pred.with_columns(pl.lit("pred").alias("type")).select(plot_obs.columns)

    plot_data = pl.concat([plot_obs, plot_pred])

    plot_new_data = plot_data.filter(
        pl.col("time_end").is_between(
            pl.col("forecast_start").min(),
            pl.col("forecast_start").max(),
            "both",
        )
    )

    obs = (
        alt.Chart(plot_new_data)
        .mark_point(color="black", filled=True)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("estimate:Q"),
        )
        .transform_filter(datum.type == "obs")
    )

    interval = (
        alt.Chart(plot_new_data)
        .transform_filter(datum.type == "pred")
        .transform_quantile(
            "estimate",
            probs=[0.0025, 0.5, 0.975],
            as_=["quantile", "estimate"],
            groupby=["time_end", "season", "forecast_start", "model"],
        )
        .mark_line()
        .encode(x="time_end:T", y="estimate:Q", color="quantile:N")
    )

    return alt.layer(interval, obs, data=plot_new_data).facet(
        column="model", row="forecast_start"
    )


def plot_score(scores: pl.DataFrame):
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
    chart object
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
    p.add_argument("--score", help="evaluation scores")
    p.add_argument("--proj_output", help="png file of projection plot")
    p.add_argument("--score_output", help="png file of score plot")
    p.add_argument("--summary_output", help="png file of mean and 95% CI plot")
    args = p.parse_args()

    pred = pl.read_parquet(args.pred)
    data = pl.read_parquet(args.obs)
    plot_individual_projections(data, pred).save(args.proj_output)
    plot_summary(data, pred).save(args.summary_output)

    # score = pl.read_parquet(args.score)
    # plot_score(score).save(args.score_output)
