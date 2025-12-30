import argparse

import polars as pl
import yaml

import iup.eval


def eval_all_forecasts(
    data: pl.DataFrame, pred: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """
    Calculates the evaluation metrics selected by config, by model and forecast start date.
    -----------------------
    Arguments:
    data:
        observed data with at least "time_end" and "estimate" columns
    pred:
        forecast data as sample distribution with at least "time_end", "sample_id", "model", "forecast_start" and "estimate",
    config:
        config file to specify the expected quantile from the sample distribution and evaluation metrics to calculate

    Returns:
        A pl.DataFrame with score name and score values, grouped by model, forecast start, quantile, and possibly other grouping factors
    """
    forecast_starts = pred["forecast_start"].unique()
    score_funs = [getattr(iup.eval, fun_name) for fun_name in config["score_funs"]]

    assert config["groups"] is not None
    cols = config["groups"] + [
        "model",
        "forecast_start",
        "score_value",
        "score_fun",
        "score_type",
    ]

    all_scores = pl.DataFrame()

    for forecast_start in forecast_starts:
        # get a fit score
        fit_data = data.filter(pl.col("time_end") <= forecast_start)
        fit_pred = pred.filter(pl.col("time_end") <= forecast_start)

        fc_data = data.filter(pl.col("time_end") > forecast_start)
        fc_pred = pred.filter(pl.col("time_end") > forecast_start)

        for score_fun in score_funs:
            fit_scores = (
                score_fun(
                    obs=fit_data, pred=fit_pred, grouping_factors=config["groups"]
                )
                .with_columns(score_type=pl.lit("fit"))
                .select(cols)
            )

            fc_scores = (
                score_fun(obs=fc_data, pred=fc_pred, grouping_factors=config["groups"])
                .with_columns(score_type=pl.lit("forecast"))
                .select(cols)
            )

            all_scores = pl.concat([all_scores, fit_scores, fc_scores])

    return all_scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", required=True)
    p.add_argument("--data", help="observed data", required=True)
    p.add_argument("--forecasts", help="forecasts parquet", required=True)
    p.add_argument("--output", help="output scores parquet", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pred = pl.read_parquet(args.forecasts)
    data = pl.read_parquet(args.data)

    eval_all_forecasts(data, pred, config).write_parquet(args.output)
