import argparse

import altair as alt
import polars as pl

import iup


def plot_projections(obs: pl.DataFrame, pred: pl.DataFrame):
    """
    Save a multiple-grid graph with the comparison between the observed uptake and the prediction,
    initiated over the season.

    Arguments:
    --------------
    obs: polars.Dataframe
        The observed uptake data frame, indicating the cumulative vaccine uptake as of `time_end`.
    pred: polars.Dataframe
        The predicted daily uptake, differed by forecast date, must include columns
        `forecast_start` and `estimate`.

    Return:
    -------------
    chart object
    """

    # input check
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    assert obs["geography"].unique().to_list() == ["nation"], "Geography is not unique"
    assert pred["geography"].unique().to_list() == ["nation"], "Geography is not unique"

    # get every model/forecast date combo
    models_forecasts = pred.select(["model", "forecast_start"]).unique()

    # for every model and forecast date, merge in the observed value
    # column "type" will be either "obs" or "pred"
    plot_obs = (
        obs.join(models_forecasts, how="cross")
        .with_columns(
            type=pl.lit("obs"),
            season=pl.col("time_end").pipe(iup.UptakeData.date_to_season),
        )
        .select(["type", "model", "forecast_start", "time_end", "estimate", "season"])
        .filter(pl.col("season").is_in(pred["season"].unique()))
    )

    plot_pred = pred.with_columns(pl.lit("pred").alias("type")).select(
        ["type", "model", "forecast_start", "time_end", "estimate", "season"]
    )

    plot_data = pl.concat([plot_obs, plot_pred])

    return (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            alt.X("time_end:T", axis=alt.Axis(format="%Y-%m", tickCount="month")),
            alt.Y("estimate:Q"),
            alt.Column("model"),
            alt.Row("forecast_start:T"),
            alt.Color("type"),
            alt.Detail("season"),
        )
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
    args = p.parse_args()

    pred = pl.read_parquet(args.pred)
    data = pl.read_parquet(args.obs)
    plot_projections(data, pred).save(args.proj_output)

    score = pl.read_parquet(args.score)
    plot_score(score).save(args.score_output)
