import datetime as dt
from typing import Callable, Dict, List

import polars as pl

from iup import CumulativeUptakeData, QuantileForecast


###### evaluation metrics #####
def check_date_match(
    data: CumulativeUptakeData, pred: QuantileForecast, groups: List[str] | None
):
    """
    Check the dates between data and pred.
    Dates must be 1-on-1 equal and no duplicate.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be CumulativeUptakeData
    pred:
        The forecast made by model. Can be QuantileForecast or PointForecast
    groups:
        A list of grouping factors

    Return
    Error if conditions fail to meet.

    """
    # sort data and pred by date #
    groups_and_time = ["time_end"] + groups if groups is not None else ["time_end"]

    data = CumulativeUptakeData(data.sort(groups_and_time))
    pred = QuantileForecast(pred.sort(groups_and_time))

    if groups is not None:
        assert [
            (
                data.filter(pl.col(group) == group_unique)["time_end"]
                == pred.filter(pl.col(group) == group_unique)["time_end"]
            ).all()
            for group in groups
            for group_unique in data[group].unique()
        ]
    else:
        assert data.shape[0] == pred.shape[0], (
            "The forecast and the data should have the same forecast dates"
        )

    if groups is not None:
        assert not (
            any(
                [
                    any(
                        data.filter(pl.col(group) == group_unique)
                        .select("time_end")
                        .is_duplicated()
                    )
                    for group in groups
                    for group_unique in data[group].unique()
                ]
            )
        )
    else:
        assert not (any(data["time_end"].is_duplicated())), (
            "Duplicated dates are found in data and prediction."
        )


def summarize_score(
    data: CumulativeUptakeData,
    pred: QuantileForecast,
    groups: List[str] | None,
    score_funs: Dict[str, Callable],
) -> pl.DataFrame:
    """
    Calculate score between observed data and forecast.
    ----------------------

    Parameters
    data:
        The observed data used for modeling. Should be CumulativeUptakeData
    pred:
        The forecast made by model. Can be QuantileForecast or PointForecast
    groups:
        A list of grouping factors, specified in config file.
    score_funs:
        A dictionary of scoring functions. The key is the name of the score, and the value
        is the scoring function.

    Return
    A pl.DataFrame of scores with information including score name and score values, grouped by quantile, forecast

    """

    check_date_match(data, pred, groups)
    assert isinstance(data, CumulativeUptakeData)
    assert isinstance(pred, QuantileForecast)

    assert len(pred["quantile"].unique()) == 1, (
        "The prediction should only have one quantile."
    )

    if groups is None:
        columns_to_join = ["time_end"]
    else:
        columns_to_join = ["time_end"] + groups

    joined_df = data.join(pred, on=columns_to_join, how="inner", validate="1:1").rename(
        {"estimate": "data", "estimate_right": "pred"}
    )

    all_scores = pl.DataFrame()
    for score_name in score_funs:
        score = joined_df.select(
            quantile=pl.col("quantile").first(),
            forecast_start=pl.col("time_end").min(),
            forecast_end=pl.col("time_end").max(),
            score_name=pl.lit(score_name),
            score_value=score_funs[score_name](pl.col("data"), pl.col("pred")),
        ).filter(pl.col("score_value").is_not_null())
        all_scores = pl.concat([all_scores, score])

    return all_scores


def mspe(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Calculate Mean Squared Prediction Error with polars column expression
    ---------------------
    Arguments:
    x: either observed data or predictions
    y: either observed data or predictions
    Return:
        Mean Squared Prediction Error as a polars column expression

    """
    return ((x - y) ** 2).mean()


def abs_diff(
    selected_date: dt.date, date_col: pl.Expr
) -> Callable[[pl.Expr, pl.Expr], pl.Expr]:
    """
    Generate a function that calculates the absolute difference between
    observed data and prediction on a certain date.
    ----------------------
    Arguments:
    selected_date: a datetime date object to specify which date to do the calculation
    date_col: a polars column expression used to select the date

    Return:
        A function that takes two polars column expressions to do the calculation.
    """

    lit_date = pl.lit(selected_date)

    def f(x: pl.Expr, y: pl.Expr) -> pl.Expr:
        """
        Calculate the absolute difference between two polars column expressions
        -----------------------
        Arguments:
        x: either observed data or predictions
        y: either observed data or predictions

        Return:
        A polars column expression that returns the absolute difference at the certain date, otherwise None
        """
        return pl.when(date_col == lit_date).then((x - y).abs()).otherwise(None)

    return f
