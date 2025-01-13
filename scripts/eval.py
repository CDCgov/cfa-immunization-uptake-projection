import argparse

import polars as pl
import yaml

import iup
from iup import eval


def eval_all_forecasts(test, pred, config):
    """Evaluate the forecasts for all models, all forecast ends, and all scores"""
    score_names = config["score_funs"]
    model_names = pred["model"].unique()
    forecast_starts = pred["forecast_start"].unique()

    # only 'incident' type is evaluated #
    incident_pred = pred.filter(pl.col("estimate_type") == "incident").with_columns(
        quantile=0.5
    )
    # This step is arbitrary, but it is necessary to pass PointForecast validation #

    all_scores = pl.DataFrame()

    for score_name in score_names:
        score_fun = getattr(eval, score_name)

        for model in model_names:
            for forecast_start in forecast_starts:
                pred_data = incident_pred.filter(
                    pl.col("model") == model, pl.col("forecast_start") == forecast_start
                )

                assert (pred_data["forecast_start"] == test["time_end"].min()).all()

                test = iup.IncidentUptakeData(test)
                pred_data = iup.PointForecast(pred_data)

                score = eval.score(test, pred_data, score_fun)
                score = score.with_columns(score_fun=score_name, model=model)

                all_scores = pl.concat([all_scores, score])

    return all_scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pred_data = pl.scan_parquet(args.pred).collect()
    obs_data = pl.scan_parquet(args.obs).collect()
    obs_data = obs_data.filter(pl.col("estimate_type") == "incident")

    # ensure the same test data is used for all models
    test_data = iup.IncidentUptakeData.split_train_test(
        obs_data, config["timeframe"]["start"], "test"
    ).filter(pl.col("time_end") <= config["timeframe"]["end"])

    test_data = iup.IncidentUptakeData(test_data)

    all_scores = eval_all_forecasts(test_data, pred_data, config)
    all_scores.write_parquet(args.output)
