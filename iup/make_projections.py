import polars as pl
import datetime as dt
import numpy as np


def make_projections(
    data,
    model,
    start_date,
    end_date,
    interval,
    init_elapsed,
    init_interval,
    init_incident,
    init_cumulative,
):
    """
    Uses a linear regression model to project per-day incident uptake.

    Parameters
    ----------
    data : DataFrame
        Uptake data used to fit the projection model,
        as imported and formatted by get_uptake_data.py
    model: LinearRegression
        Linear regression model object from scikit-learn
        that is fit to the uptake data
    start_date: string
        Date for which current uptake information is known,
        in %Y-%m-%d format
    end_date: string
        Date for through which uptake projections are desired,
        in %Y-%m-%d format
    interval: string
        Time interval between projection days, as specified in
        polars.date_range()
    init_elapsed: number
        Number of days since rollout on the start_date
    init_interval: number
        Number of days since the previous report on the start_date
    init_incident: number
        Incident uptake as reported on the start_date
    init_cumulative: number
        Cumulative uptake as reported on the start_date

    Details
    -------

    THIS DOES NOT YET SUPPORT YEARLY VARIATION, MULTIPLE STATES,
    OR DIFFERENT MODEL STRUCTURES.
    The data used to fit the projection model are provided and trimmed
    (by removing the first 1-2 rows) exactly as in build_projection_model.py,
    to extract the exact standardization constants used in the model.
    An output DataFrame is built to mirror the structure of the training data,
    but for the dates on which projection is desired. Projection is performed
    one date at a time sequentially, which allows error and/or uncertainty
    to propagate. Uncertainty is not included here, because the ordinary least
    squares prediction intervals are neither forthcoming from scikit-learn, nor
    do they generalize to the Bayesian projection model fitting we ultimately
    desire. Projections for daily average uptake are transformed to incident
    and cumulative uptake.

    Returns
    -------
    DataFrame
        Projected incident and cumulative uptake.
    """
    if ((data["interval"] - data["interval"].mean()) / data["interval"].std())[0] > 1:
        train_data = data.slice(2, data.height - 1)
    else:
        train_data = data.slice(1, data.height - 1)

    previous_mn = train_data["previous"].mean()
    previous_sd = train_data["previous"].std()
    elapsed_mn = train_data["elapsed"].mean()
    elapsed_sd = train_data["elapsed"].std()
    daily_mn = train_data["daily"].mean()
    daily_sd = train_data["daily"].std()

    output = (
        pl.date_range(
            start=dt.datetime.strptime(start_date, "%Y-%m-%d"),
            end=dt.datetime.strptime(end_date, "%Y-%m-%d"),
            interval=interval,
            eager=True,
        )
        .alias("date")
        .to_frame()
        .with_columns(
            state=pl.lit(train_data.select(pl.col("state").unique())),
            elapsed=(
                (pl.col("date") - dt.datetime.strptime(start_date, "%Y-%m-%d")).cast(
                    pl.Float64
                )
                / (1000000 * 60 * 60 * 24)
            )
            + init_elapsed,
            season=pl.col("date").dt.year()
            + pl.when(pl.col("date").dt.month() < 7).then(-1).otherwise(0),
        )
        .with_columns(interval=pl.col("elapsed").diff().fill_null(init_interval))
    )

    proj = np.zeros((output.shape[0]))
    proj[0] = init_incident / init_interval

    for i in range(proj.shape[0] - 1):
        x = np.column_stack(
            (
                (proj[i] - previous_mn) / previous_sd,
                (output["elapsed"][i + 1] - elapsed_mn) / elapsed_sd,
            )
        )
        x = np.insert(x, 2, np.array((x[:, 0] * x[:, 1])), axis=1)
        y = model.predict(x)
        proj[i + 1] = y[(0, 0)] * daily_sd + daily_mn

    output = (
        output.with_columns(daily=pl.Series(proj))
        .with_columns(incident=pl.col("daily") * pl.col("interval"))
        .with_columns(
            cumulative=pl.col("incident").cum_sum() + init_cumulative - init_incident
        )
    )

    return output
