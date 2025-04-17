import argparse

import polars as pl
import yaml

import iup
from iup import eval


def eval_all_forecasts(
    data: pl.DataFrame, pred: pl.DataFrame, config: dict
) -> pl.DataFrame:
    """ "
    Calculates the evaluation metrics selected by config, by model and forecast start date.
    -----------------------
    Arguments:
    data:
        observed data with at least "time_end" and "estimate" columns
    pred:
        forecast data as sample distribution with at least "time_end", "sample_id", "model", "forecast_start" and "estimate"
    config:
        config file to specify the expected quantile from the sample distribution and evaluation metrics to calculate

    Returns:
        A pl.DataFrame with score name and score values, grouped by model, forecast start and quantile
    """
    model_names = pred["model"].unique()
    forecast_starts = pred["forecast_start"].unique()

    all_scores = pl.DataFrame()

    for model in model_names:
        for forecast_start in forecast_starts:
            incident_pred = iup.SampleForecast(
                iup.CumulativeUptakeData(
                    pred.filter(
                        pl.col("model") == model,
                        pl.col("forecast_start") == forecast_start,
                    )
                ).to_incident(config["data"]["groups"])
            )
            test = iup.CumulativeUptakeData(
                data.filter(
                    pl.col("time_end") >= forecast_start,
                    pl.col("time_end") <= config["forecast_timeframe"]["end"],
                )
            ).to_incident(config["data"]["groups"])

            # 1. Convert sample forecast pred into quantiles
            assert config["scores"]["quantiles"] is not None, (
                "Quantiles of posterior prediction distribution must be specified in the config file."
            )
            for quantile in config["scores"]["quantiles"]:
                summary_pred = incident_pred.group_by("time_end").agg(
                    pl.col("estimate").quantile(quantile)
                )

                assert summary_pred.shape[0] == test.shape[0], (
                    "The forecast and the test data do not have the same number of dates."
                )

                # 2. Calculate the score for each quantile
                score_df = test.join(
                    summary_pred, on="time_end", how="inner", validate="1:1"
                ).rename({"estimate": "data", "estimate_right": "pred"})

                if config["scores"]["difference_by_date"] is not None:
                    score_funcs = {
                        f"{eval.abs_diff.__name__}_{date}": eval.abs_diff(
                            date, pl.col("time_end")
                        )
                        for date in config["scores"]["difference_by_date"]
                    }

                score_funcs[config["scores"]["others"]] = getattr(
                    eval, config["scores"]["others"]
                )

                for score_name in score_funcs:
                    score_fun = score_funcs[score_name]
                    score = score_df.select(
                        model=pl.lit(model),
                        forecast_start=pl.col("time_end").min(),
                        forecast_end=pl.col("time_end").max(),
                        quantile=quantile,
                        score_name=pl.lit(score_name),
                        score_value=score_fun(pl.col("data"), pl.col("pred")),
                    ).filter(pl.col("score_value").is_not_null())

                    all_scores = pl.concat([all_scores, score])

    return all_scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pred = pl.read_parquet(args.pred)
    data = pl.read_parquet(args.obs)

    if config["evaluation_timeframe"]["interval"] is not None:
        eval_all_forecasts(data, pred, config).write_parquet(args.output)
