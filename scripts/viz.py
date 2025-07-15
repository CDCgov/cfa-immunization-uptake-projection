from typing import Any, Dict, List

import altair as alt
import numpy as np
import polars as pl
import streamlit as st
import yaml


def app():
    st.title("Vaccine Uptake Forecasts")

    # data prepare
    # load data
    data = load_data()
    obs = data["observed"]
    pred = data["forecasts"]

    # load config
    config = load_config()

    # multiple tabs
    tab1, tab2, tab3 = st.tabs(["Trajectories", "Summary", "Evaluation"])

    with tab1:
        plot_trajectories(obs, pred, config)

    with tab2:
        plot_summary(obs=obs, pred=pred, config=config)

    with tab3:
        plot_evaluation(load_scores(), config)


def plot_trajectories(obs: pl.DataFrame, pred: pl.DataFrame, config: Dict[str, Any]):
    """
    Plot the individual trajectories of the forecasts with data, with user options to select
    the dimensions to group the data, including: column and row. Other grouping factors that
    haven't been selected will be used to filter the data.
    ----------------------
    Arguments
    obs: pl.DataFrame
        The observed data.
    pred: pl.DataFrame
        The forecast made by model.
    config: Dict[str, Any]
        The configuration yaml file.

    """

    # set up plot encodings
    encodings = {
        "x": alt.X("time_end:T", title="Observation date"),
        "y": alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        "color": alt.Color("sample_id:N", title="Trajectories"),
    }

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["model", "forecast_start"] + config["data"]["groups"]
    default_channels = {
        "column": ("Column", "forecast_start"),
        "row": ("Row", "model"),
    }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(
            f"{channel} by",
            options=options,
            index=index,
            key=f"{channel}_{idx}_trajectories",
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for dim in dimensions:
        filter_val = st.selectbox(dim, options=pred[dim].unique().sort().to_list())
        pred = pred.filter(pl.col(dim) == pl.lit(filter_val))

    # how many trajectories to show?
    n_samples = st.number_input("Number of forecasts", value=3, min_value=1)

    # draw indices of trajectories randomly #
    rng = np.random.default_rng()

    selected_ids = rng.integers(
        low=pred["sample_id"].cast(pl.Int64).min(),
        high=pred["sample_id"].cast(pl.Int64).max() + 1,
        size=n_samples,
    )

    pred = pred.filter(pl.col("sample_id").cast(pl.Int32).is_in(selected_ids))

    # merge observed data with prediction by the combination of models and forecast starts
    model_forecast_starts = pred.select(["model", "forecast_start"]).unique()
    plot_obs = obs.join(model_forecast_starts, how="cross").filter(
        pl.col(factor).is_in(pred[factor].unique())
        for factor in config["data"]["groups"]
    )

    groupings = ["model", "forecast_start", "time_end"] + config["data"]["groups"]

    data = pred.join(plot_obs, on=groupings).rename({"estimate_right": "observed"})

    obs_chart = alt.Chart(data).mark_point().encode(x="time_end:T", y="observed:Q")

    pred_chart = alt.Chart(data).mark_line().encode(x="time_end:T", y="estimate:Q")

    chart = layer_with_facets([obs_chart, pred_chart], encodings)

    st.altair_chart(chart, use_container_width=True)


def plot_summary(obs: pl.DataFrame, pred: pl.DataFrame, config: Dict[str, Any]):
    """
    Plot the 95% PI with mean estimate of forecasts with data, with user options to select
    the dimensions to group the data, including: row, column, and color. Other grouping
    factors that haven't been selected will be used to filter the data.
    ----------------------
    Arguments
    obs: pl.DataFrame
        The observed data.
    pred: pl.DataFrame
        The forecast made by model.
    config: Dict[str, Any]
        The configuration yaml file.
    """
    encodings = {}

    # data process: merge observed data with prediction by combinations of model and forecast start #
    forecast_starts = pred.select(["model", "forecast_start"]).unique()
    plot_obs = obs.join(forecast_starts, how="cross").filter(
        pl.col(factor).is_in(pred[factor].unique())
        for factor in config["data"]["groups"]
    )

    # summarize sample predictions by grouping factors #
    groups_to_include = ["model", "forecast_start", "time_end"] + config["data"][
        "groups"
    ]

    plot_pred = (
        pred.group_by(groups_to_include)
        .agg(
            lower=pl.col("estimate").quantile(
                config["forecast_plots"]["interval"]["lower"]
            ),
            upper=pl.col("estimate").quantile(
                config["forecast_plots"]["interval"]["upper"]
            ),
            mean=pl.col("estimate").mean(),
        )
        .sort("time_end")
    )

    data = plot_pred.join(plot_obs, on=groups_to_include)

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["model", "forecast_start"] + config["data"]["groups"]

    if "season" in dimensions:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "forecast_start"),
            "row": ("Row", "season"),
        }
    else:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "forecast_start"),
            "row": ("Row", "None"),
        }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(
            f"{channel} by",
            options=options,
            index=index,
            key=f"{channel}_{idx}_summary",
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for idx, dim in enumerate(dimensions):
        filter_val = st.selectbox(
            dim,
            options=plot_pred[dim].unique().sort().to_list(),
            key=f"{dim}_{idx}_summary",
        )

        data = data.filter(pl.col(dim) == pl.lit(filter_val))

    obs_chart = (
        alt.Chart(data)
        .mark_point(color="black", filled=True)
        .encode(
            alt.X("time_end:T", title="Observation date"),
            alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        )
    )

    pred_chart = (
        alt.Chart(data)
        .mark_line(color="grey")
        .encode(
            alt.X("time_end:T", title="Observation date"),
            alt.Y("mean:Q", title="Cumulative uptake estimate"),
        )
    )

    interval_chart = (
        alt.Chart(data)
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

    chart_list = [interval_chart, pred_chart, obs_chart]
    chart = layer_with_facets(chart_list, encodings)

    st.altair_chart(chart, use_container_width=True)


def plot_evaluation(scores: pl.DataFrame, config: Dict[str, Any]):
    """
    Plot the evaluation scores over forecast start. User can select
    the dimensions to group the data, including: row, column, and color. Other grouping
    factors that haven't been selected will be used to filter the data.
    ----------------------
    Arguments
    scores: pl.DataFrame
        The evaluation scores of the forecasts.
    config: Dict[str, Any]
        The configuration yaml file.
    """

    encodings = {
        "x": alt.X("forecast_start:T", title="Forecast start date"),
        "y": alt.Y("score_value:Q", title="Score value"),
    }

    score_names = scores["score_name"].unique()

    if "mspe" in score_names:
        score_dict = {
            "mspe": "Mean Squared Prediction Error",
        }
    else:
        score_dict = {}

    for name in score_names:
        if name.startswith("abs_diff_"):
            score_dict[name] = "Absolute difference at " + name[len("abs_diff_") :]

    # every score name should have a label for the plot
    assert set(score_names).issubset(score_dict.keys())

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["model", "score_name"] + config["data"]["groups"]
    if "season" in dimensions:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "score_name"),
            "row": ("Row", "season"),
        }
    else:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "score_name"),
            "row": ("Row", "None"),
        }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(
            f"{channel} by", options=options, index=index, key=f"{channel}_{idx}_eval"
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    for idx, dim in enumerate(dimensions):
        filter_val = st.selectbox(
            dim, options=scores[dim].unique().sort().to_list(), key=f"{dim}_{idx}_eval"
        )
        plot_score = scores.filter(pl.col(dim) == pl.lit(filter_val))

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
def layer_with_facets(charts: List, encodings: Dict):
    """
    Because alt.layer.facet() only takes row and column and .encode() only takes color,
    this function makes sure correct arguments fall into correct command.
    ----------------------
    Arguments
    data: pl.DataFrame
        The data to be plotted.
    charts: list of alt.Chart
        The charts to be layered.
    encodings: dict
        The encodings to be applied to the charts, including row, column, and other encodings.
    """

    row_enc = encodings["row"]
    col_enc = encodings["column"]
    other_encs = {k: v for k, v in encodings.items() if k not in ["row", "column"]}
    print(other_encs)

    layered = alt.layer(*charts)

    layered = layered.encode(**other_encs)

    return layered.facet(row=row_enc, column=col_enc)


@st.cache_data
def load_data():
    return {
        "forecasts": pl.read_parquet("output/forecasts/tables/forecasts.parquet"),
        "observed": pl.read_parquet("output/data/nis_raw_flu.parquet"),
    }


@st.cache_data
def load_scores():
    return pl.read_parquet("output/scores/test/scores.parquet")


@st.cache_data
def load_config():
    return yaml.safe_load(open("scripts/config.yaml"))


if __name__ == "__main__":
    app()
