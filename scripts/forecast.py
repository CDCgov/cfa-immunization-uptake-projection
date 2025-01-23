import argparse
import datetime
import warnings
from typing import List

import polars as pl
import yaml

import iup.models


def run_forecast(
    dataset_path: str,
    model_name: str,
    forecast_date: datetime.date,
    target_dates: List[datetime.date],
    grouping_factors,
    output_path: str,
) -> pl.DataFrame:
    """Run a single model for a single forecast date"""
    # check that target dates are after the forecast date
    warnings.warn("not implemented")

    # get model object from name
    model = getattr(iup.models, model_name)
    assert issubclass(model, iup.models.UptakeModel)

    # get data to use for forecast
    data = pl.scan_parquet(dataset_path)
    training_data = iup.IncidentUptakeData.split_train_test(
        data, forecast_date, "train"
    )

    # check that target dates are not present in the training data
    warnings.warn("not implemented")

    # fit model and run predictions
    fit = model().fit(training_data)
    pred = fit.predict(target_dates, grouping_factors)

    # write output
    pred.write_parquet(output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="input dataset")
    p.add_argument("--model", help="model to forecast with")
    p.add_argument("--forecast_date", help="forecast date")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_data = iup.CumulativeUptakeData(pl.scan_parquet(args.input).collect())

    target_dates = None
    warnings.warn("need to figure out target dates")
    grouping_factors = None
    warnings.warn("need to figure out grouping factors")

    run_forecast(
        dataset_path=args.input,
        model_name=args.model,
        forecast_date=datetime.date.fromisoformat(args.forecast_date),
        target_dates=target_dates,
        grouping_factors=grouping_factors,
        output_path=args.output,
    )
