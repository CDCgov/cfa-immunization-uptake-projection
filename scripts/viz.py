import altair as alt
import polars as pl
import streamlit as st


def app():
    st.title("Vaccine Uptake Forecasts")

    # data prepare
    # load data
    data = load_data()
    obs = data["observed"]
    pred = data["forecasts"]

    # clean data
    forecast_data = (
        data["forecasts"]
        .with_columns(
            pl.col("forecast_start").dt.truncate("1d"),
            pl.col("sample_id").cast(pl.Int32),
        )
        .drop("forecast_end")
    )

    # get all forecast start dates and cross them with the observed data, as
    # if the observed data were a model
    forecast_starts = forecast_data.select(["season", "forecast_start"]).unique()

    obs_data = (
        data["observed"]
        .with_columns(sample_id=0, model=pl.lit("observed"))
        .join(forecast_starts, on=["season"], how="inner")
        .select(forecast_data.columns)
    )

    data = pl.concat([forecast_data, obs_data], how="vertical")
    data = data.filter(pl.col("geography").is_in(["Alabama", "Alaska"]))

    # multiple tabs
    tab1, tab2, tab3 = st.tabs(["Trajectories", "Summary", "Evaluation"])

    with tab1:
        plot_trajectories(data)

    with tab2:
        plot_summary(obs=obs, pred=pred)
    with tab3:
        plot_evaluation(load_scores())


def plot_trajectories(data):
    # set up plot encodings
    encodings = [
        alt.X("time_end:T", title="Observation date"),
        alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        alt.Detail("sample_id", title="Trajectory"),
    ]

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["season", "model", "forecast_start", "geography"]
    default_channels = [
        ("Color", "model"),
        ("Column", "forecast_start"),
        ("Row", "season"),
    ]

    for channel, default_dim in default_channels:
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(f"{channel} by", options=options, index=index)

        if dim != "None":
            dimensions.remove(dim)
            encodings.append(getattr(alt, channel)(dim))

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for dim in dimensions:
        filter_val = st.selectbox(dim, options=data[dim].unique().sort().to_list())
        data = data.filter(pl.col(dim) == pl.lit(filter_val))

    # how many trajectories to show?
    n_samples = st.number_input("Number of forecasts", value=3, min_value=1)
    data = data.filter(
        pl.col("sample_id").cast(pl.Int32) <= n_samples,
    )

    chart = alt.Chart(data).mark_line().encode(*encodings)

    st.altair_chart(chart, use_container_width=True)


def plot_summary(obs: pl.DataFrame, pred: pl.DataFrame):
    encodings = {}

    # reformat the observed data to be ready to join with pred data #
    models_forecasts = pred.select(["model", "forecast_start"]).unique()
    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique())
    )

    plot_pred = (
        pred.group_by(["model", "forecast_start", "time_end", "season", "geography"])
        .agg(
            lower=pl.col("estimate").quantile(0.025),
            upper=pl.col("estimate").quantile(0.975),
            mean=pl.col("estimate").mean(),
        )
        .sort("time_end")
    )

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["season", "model", "forecast_start", "geography"]
    default_channels = {
        "color": ("Color", "model"),
        "column": ("Column", "forecast_start"),
        "row": ("Row", "season"),
    }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(
            f"{channel} by", options=options, index=index, key=f"{channel}_{idx}"
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for idx, dim in enumerate(dimensions):
        filter_val = st.selectbox(
            dim, options=plot_pred[dim].unique().sort().to_list(), key=f"{dim}_{idx}"
        )
        plot_pred = plot_pred.filter(pl.col(dim) == pl.lit(filter_val))

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point()
        .encode(
            alt.X("time_end:T", title="Observation date"),
            alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        )
    )

    pred_chart = (
        alt.Chart()
        .mark_line(color="grey")
        .encode(
            alt.X("time_end:T", title="Observation date"),
            alt.Y("mean:Q", title="Cumulative uptake estimate"),
        )
    )

    interval_chart = (
        alt.Chart()
        .mark_area(opacity=0.3)
        .encode(
            alt.X(
                "time_end:T",
                title="Observation date",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            y="lower:Q",
            y2="upper:Q",
        )
    )

    chart_lists = [interval_chart, pred_chart, obs_chart]

    chart = layer_with_facets(plot_pred, *chart_lists, **encodings)

    st.altair_chart(chart, use_container_width=True)


def plot_evaluation(scores: pl.DataFrame):
    encodings = {
        "x": alt.X("forecast_start:T", title="Forecast start date"),
        "y": alt.Y("score_value:Q", title="Score value"),
    }

    score_names = scores["score_name"].unique()

    score_dict = {
        "mspe": "Mean Squared Prediction Error",
    }

    for name in score_names:
        if name.startswith("abs_diff_"):
            score_dict[name] = "Absolute differenece at " + name[len("abs_diff_") :]

    # every score name should have a label for the plot
    assert set(score_names).issubset(score_dict.keys())

    # only plot the median quantile #
    plot_score = scores.filter(pl.col("quantile") == 0.5)

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["season", "model", "geography", "score_name"]
    default_channels = {
        "color": ("Color", "model"),
        "column": ("Column", "geography"),
        "row": ("Row", "season"),
    }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(
            f"{channel} by", options=options, index=index, key=f"{channel}_{idx}"
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for idx, dim in enumerate(dimensions):
        filter_val = st.selectbox(
            dim, options=plot_score[dim].unique().sort().to_list(), key=f"{dim}_{idx}"
        )
        plot_score = plot_score.filter(pl.col(dim) == pl.lit(filter_val))

    plot_score = plot_score.with_columns(
        pl.col("score_name").replace_strict(score_dict)
    )

    chart = (
        alt.Chart(plot_score)
        .mark_point()
        .encode(**encodings)
        .resolve_scale(y="independent")
    )

    st.altair_chart(chart, use_container_width=True)


## helper: feed correct argument to altair ##
def layer_with_facets(data, *charts, **encodings):
    """Because alt.layer.facet() only takes row and column and .encode() only takes color,
    this function makes sure correct arguments fall into correct command.
    """

    row_enc = encodings["row"]
    col_enc = encodings["column"]
    other_encs = {k: v for k, v in encodings.items() if k not in ["row", "column"]}

    layered = alt.layer(*charts, data=data)

    layered = layered.encode(**other_encs)

    return layered.facet(row=row_enc, column=col_enc)


@st.cache_data
def load_data():
    return {
        "forecasts": pl.read_parquet("output/forecasts/tables/forecasts.parquet"),
        "observed": pl.read_parquet("data/nis_raw_flu.parquet"),
    }


@st.cache_data
def load_scores():
    return pl.read_parquet("output/scores/tables/scores.parquet")


if __name__ == "__main__":
    app()
