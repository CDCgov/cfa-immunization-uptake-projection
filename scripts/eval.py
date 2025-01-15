import argparse

import polars as pl
import yaml

import iup
from iup import eval


def eval_all_forecasts(data, pred, config):
    """Evaluate the forecasts for all models, all forecast ends, and all scores"""
    score_names = config["score_funs"]
    model_names = pred["model"].unique()
    forecast_starts = pred["forecast_start"].unique()

    all_scores = pl.DataFrame()

    for score_name in score_names:
        score_fun = getattr(eval, score_name)

        for model in model_names:
            for forecast_start in forecast_starts:
                this_pred = pred.filter(
                    pl.col("model") == model, pl.col("forecast_start") == forecast_start
                )

                # convert cumulative predictions to incident predictions given certain forecast period and model #
                incident_pred = iup.CumulativeUptakeData(this_pred).to_incident(
                    config["data"]["groups"]
                )
                # This step is arbitrary, but it is necessary to pass PointForecast validation #
                incident_pred = incident_pred.with_columns(quantile=0.5)
                incident_pred = iup.PointForecast(incident_pred)

                test = data.filter(
                    pl.col("time_end") >= forecast_start,
                    pl.col("time_end") < config["timeframe"]["end"],
                )

                assert (incident_pred["forecast_start"] == test["time_end"].min()).all()

                test = iup.IncidentUptakeData(test)

                score = eval.score(test, incident_pred, score_fun)
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

    # ensure the same incident test data is used for all models
    obs_data = iup.CumulativeUptakeData.to_incident(config["groups"])

    all_scores = eval_all_forecasts(obs_data, pred_data, config)
    all_scores.write_parquet(args.output)
