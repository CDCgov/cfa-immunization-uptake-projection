import altair as alt
import polars as pl
import argparse
import yaml


def plot_projections(obs, pred, config):
    """
    Save a multiple-grid graph with the comparison between the observed uptake and the prediction,
    initiated over the season.

    Arguments:
    --------------
    obs: polars.Dataframe
        The observed uptake data frame, indicating the cumulative vaccine uptake as of `time_end`.
    pred: polars.Dataframe
        The predicted daily uptake, differed by forecast date, must include columns `forecast_start` and `estimate`.
    config: dict
        config.yaml to define the number of columns in the graph, the saving path and the saving name.

    Return:
    -------------
    None. The graph is saved.

    """

    # input check #
    if "time_end" not in obs.columns or "estimate" not in obs.columns:
        ValueError("'time_end' or 'estimate' is missing from obs.")

    # plot weekly initiated prediction #
    time_axis = alt.Axis(format="%Y-%m", tickCount="month")

    obs_chart = (
        alt.Chart(obs)
        .mark_circle(color="black")
        .encode(alt.X("time_end", axis=time_axis), alt.Y("estimate"))
    )

    charts = []

    forecast_starts = [date for date in pred["forecast_start"].unique()]

    for forecast_start in forecast_starts:
        single_pred = pred.filter(pl.col("forecast_start") == forecast_start)

        chart = (
            alt.Chart(single_pred)
            .mark_line(color="red")
            .encode(
                x=alt.X("time_end:T", title="Date", axis=time_axis),
                y=alt.Y("estimate:Q", title="Cumulative Vaccine Uptake (%)"),
            )
            .properties(title=f"Start Date: {forecast_start}")
        )

        obs_chart = obs_chart.encode(x=chart.encoding.x)

        chart = obs_chart + chart
        chart = chart.resolve_scale(x="shared")
        charts.append(chart)

    rows = []

    for i in range(0, len(charts), config["projection_plot"]["subplot_ncols"]):
        row = alt.hconcat(
            *list(charts[i : i + config["projection_plot"]["subplot_ncols"]])
        )
        rows.append(row)

    return alt.vconcat(*rows).configure_title(fontSize=30)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--pred", help="forecast data")
    p.add_argument("--obs", help="observed data")
    p.add_argument("--output", help="output png file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pred = pl.scan_parquet(args.pred).collect()
    data = pl.scan_parquet(args.obs).collect()

    # Drop all samples and just use mean estimate, for now
    pred = pred.drop([col for col in pred.columns if "estimate_" in col])

    if config["projection_plot"]["plot"]:
        plot_projections(data, pred, config["projection_plot"]["subplot_ncols"]).save(
            args.output
        )
