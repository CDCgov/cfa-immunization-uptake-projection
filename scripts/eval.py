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
                incident_pred = iup.PointForecast(
                    iup.CumulativeUptakeData(
                        pred.filter(
                            pl.col("model") == model,
                            pl.col("forecast_start") == forecast_start,
                        )
                    )
                    .to_incident(config["data"]["groups"])
                    .with_columns(quantile=0.5)
                )

                test = iup.CumulativeUptakeData(
                    data.filter(
                        pl.col("time_end") >= forecast_start,
                        pl.col("time_end") <= config["timeframe"]["end"],
                    )
                ).to_incident(config["data"]["groups"])

                assert incident_pred["forecast_start"][0] == test["time_end"].min(), (
                    "Dates do not match between the forecast and the test data"
                )

                score = eval.score(test, incident_pred, score_fun).with_columns(
                    score_fun=pl.lit(score_name),
                    model=pl.lit(model),
                )

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

    pred = pl.scan_parquet(args.pred).collect()
    data = pl.scan_parquet(args.obs).collect()

    eval_all_forecasts(data, pred, config).write_parquet(args.output)
