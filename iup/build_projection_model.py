import polars as pl
from sklearn.linear_model import LinearRegression


def build_projection_model(data):
    """
    Fits a linear regression model to explain per-day incident uptake.

    Parameters
    ----------
    data : DataFrame
        Uptake data as imported and formatted by get_uptake_data.py

    Details
    -------

    THIS DOES NOT YET SUPPORT YEARLY VARIATION OR MULTIPLE STATES.
    The data is prepared for regression modeling in several ways.
    The first report is removed (because it has no previous incident uptake),
    and if the first report's interval is far from the average interval,
    the second report is also removed (because its previous incident uptake
    is unreliable). For each report, the model predictors - the time elapsed
    since rollout and the previous report's per-day incident uptake - are
    standardized, as is the model outcome - the per-day incident uptake.
    Finally, a linear regression is fit that explains the outcome based on
    an intercept, the individual predictors, and their interaction.

    Returns
    -------
    LinearRegression
        Fitted model of incident uptake.
    """
    if ((data["interval"] - data["interval"].mean()) / data["interval"].std())[0] > 1:
        data = data.slice(2, data.height - 1)
    else:
        data = data.slice(1, data.height - 1)

    data = data.with_columns(
        previous_std=(pl.col("previous") - pl.mean("previous")) / pl.std("previous"),
        elapsed_std=(pl.col("elapsed") - pl.mean("elapsed")) / pl.std("elapsed"),
        daily_std=(pl.col("daily") - pl.mean("daily")) / pl.std("daily"),
    )

    x = (
        data.select(["previous_std", "elapsed_std"])
        .with_columns(interact=pl.col("previous_std") * pl.col("elapsed_std"))
        .to_numpy()
    )
    y = data.select(["daily_std"]).to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    return model
