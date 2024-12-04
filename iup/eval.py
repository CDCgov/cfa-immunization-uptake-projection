from iup import IncidentUptakeData, PointForecast
import polars as pl


def get_mspe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate MSPE between data and pred
    ----------------------
    Input: data, pred in a matching and validated format
    Return: pl.DataFrame with MSPE and the forecast start date and end date

    """
    # Check the conditions for date match:
    # 1. Mutual dates must exist between data and prediction.
    assert any(
        data["date"].is_in(pred["date"])
    ), "No matched dates between data and prediction."

    # 2. There should not be any duplicated date in either data or prediction.
    common_dates = data.filter(pl.col("date").is_in(pred["date"])).select("date")

    assert (
        len(common_dates) == common_dates.n_unique()
    ), "Duplicated dates are found in data or prediction."

    joined = data.join(pred, on="date", how="inner", validate="1:1")

    start = (
        joined.filter(
            pl.col("date") == pl.col("date").min(),
        )
        .rename({"date": "forecast_start"})
        .select("forecast_start")
    )

    end = (
        joined.rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(spe=(pl.col("data") - pl.col("pred")) ** 2)
        .with_columns(
            mspe=pl.col("spe").mean(),
        )
        .filter(pl.col("date") == pl.col("date").max())
        .rename({"date": "forecast_end"})
        .select("forecast_end", "mspe")
    )

    return pl.concat([start, end], how="horizontal")


def get_mean_bias(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate Mean bias from joined data.
    Note the bias here is not the classical bias calculated from the posterior distribution.

    The bias here is defined as: at time t,
    bias = -1 if pred_t < data_t; bias = 0 if pred_t == data_t; bias = 1 if pred_t > bias_t

    mean_bias = sum of the bias across time/length of data
    -------------------------
    Input: data, pred in a matching and validated format
    Return: pl.DataFrame with mean bias and the forecast start date and end date
    """

    # Check the conditions for date match:
    # 1. Mutual dates must exist between data and prediction.
    assert any(
        data["date"].is_in(pred["date"])
    ), "No matched dates between data and prediction."

    # 2. There should not be any duplicated date in either data or prediction.
    common_dates = data.filter(pl.col("date").is_in(pred["date"])).select("date")

    assert (
        len(common_dates) == common_dates.n_unique()
    ), "Duplicated dates are found in data or prediction."

    joined = (
        data.join(pred, on="date", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(diff=(pl.col("data") - pl.col("pred")))
    )

    joined = joined.with_columns(bias=joined["diff"].sign())

    m_bias = pl.DataFrame(
        {
            "forecast_start": joined["date"].min(),
            "forecast_end": joined["date"].max(),
            "mbias": joined["bias"].sum() / joined.shape[0],
        }
    )

    return m_bias


def get_eos_abe(data: IncidentUptakeData, pred: PointForecast) -> pl.DataFrame:
    """
    Calculate the absolute error of the total uptake at the end of season between data and prediction.
    -------------------
    Input: data, pred in a matching and validated format
    Return: pl.DataFrame with absolute error in the total uptake between data and prediction
            and the forecast end date.
    """
    joined = (
        data.join(pred, on="date", how="inner", validate="1:1")
        .rename({"estimate": "data", "estimate_right": "pred"})
        .with_columns(
            cumu_data=pl.col("data").cum_sum(),
            cumu_pred=pl.col("pred").cum_sum(),
        )
        .filter(pl.col("date") == pl.col("date").max())
        .rename({"date": "forecast_end"})
    )

    abe_perc = abs(joined["cumu_data"] - joined["cumu_pred"]) / joined["cumu_data"]

    return pl.DataFrame([joined["forecast_end"], abe_perc])
