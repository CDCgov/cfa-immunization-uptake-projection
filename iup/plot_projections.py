# plot projections #
import altair as alt


def plot_projections(obs_data, pred_data_list, n_columns, pic_loc, pic_name):
    """
    Save a multiple-grid graph with the comparison between the observed uptake and the prediction,
    initiated over the season.

    Parameters:
    --------------
    obs_data: polars.Dataframe
        The observed uptake with `date` and `cumulative`, indicating the cumulative vaccine uptake as of `date`.
    pred_data_list: list containing multiple polars.Dataframe
        The predicted daily uptake, differed by initialized date, must include columns `date` and `cumulative`.
    n_columns:
        The number of columns in the graph.
    pic_loc:
        The directory to save the graph.
    pic_name:
        The name of the graph.

    Return:
    -------------
    None. The graph is saved.

    """

    # input check #
    if "time_end" not in obs_data.columns or "cumulative" not in obs_data.columns:
        ValueError("'time_end' or 'cumulative' is missing from obs_data.")

    if not isinstance(pred_data_list, list):
        ValueError("pred_data_list must be a list.")

    if (
        "time_end" not in pred_data_list[0].columns
        or "cumulative" not in pred_data_list[0].columns
    ):
        ValueError("'time_end' or 'cumulative' is missing from pred_data_list.")

    # plot weekly initiated prediction #
    time_axis = alt.Axis(format="%Y-%m", tickCount="month")

    obs = (
        alt.Chart(obs_data)
        .mark_circle(color="black")
        .encode(alt.X("time_end", axis=time_axis), alt.Y("cumulative"))
    )

    charts = []

    dates = [pred_data_list[i]["time_end"].min() for i in range(len(pred_data_list))]

    for i in range(len(pred_data_list)):
        date = dates[i].strftime("%Y-%m-%d")
        pred = pred_data_list[i]

        chart = (
            alt.Chart(pred)
            .mark_line(color="red")
            .encode(
                x=alt.X("time_end:T", title="Date", axis=time_axis),
                y=alt.Y("cumulative:Q", title="Cumulative Vaccine Uptake (%)"),
            )
            .properties(title=f"Start Date: {date}")
        )

        obs = obs.encode(x=chart.encoding.x)

        chart = obs + chart
        chart = chart.resolve_scale(x="shared")
        charts.append(chart)

    rows = []

    for i in range(0, len(charts), n_columns):
        row = alt.hconcat(*list(charts[i : i + n_columns]))
        rows.append(row)

    alt.vconcat(*rows).configure_title(fontSize=30).save(f"{pic_loc}{pic_name}")
