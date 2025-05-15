import datetime as dt
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
    rng = np.random.default_rng(seed=int(dt.datetime.now().timestamp()))

    selected_ids = rng.integers(
        low=pred["sample_id"].cast(pl.Int64).min(),
        high=pred["sample_id"].cast(pl.Int64).max() + 1,
        size=n_samples,
    )

    pred = pred.filter(pl.col("sample_id").cast(pl.Int32).is_in(selected_ids))

    # get every model/forecast date combo
    models_forecasts = pred.select(["model", "forecast_start"]).unique()

    # for every model and forecast date, merge in the observed value
    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique()),
        pl.col("geography").is_in(pred["geography"].unique()),
    )

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point(filled=True, color="black")
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("estimate:Q"),
        )
    )

    pred_chart = (
        alt.Chart(pred)
        .mark_line(opacity=0.3)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            alt.Y("estimate:Q"),
            alt.Color("sample_id:N", title="Sample ID"),
        )
    )
    chart_lists = [pred_chart, obs_chart]
    chart = layer_with_facets(pred, *chart_lists, **encodings)

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

    # reformat the observed data to be ready to join with pred data #
    models_forecasts = pred.select(["model", "forecast_start"]).unique()
    plot_obs = obs.join(models_forecasts, how="cross").filter(
        pl.col("season").is_in(pred["season"].unique())
    )

    groups_to_include = ["model", "forecast_start", "time_end"] + config["data"][
        "groups"
    ]
    plot_pred = (
        pred.group_by(groups_to_include)
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
        plot_pred = plot_pred.filter(pl.col(dim) == pl.lit(filter_val))
        plot_obs = plot_obs.filter(pl.col(dim) == pl.lit(filter_val))

    obs_chart = (
        alt.Chart(plot_obs)
        .mark_point(color="black", filled=True)
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

    score_dict = {
        "mspe": "Mean Squared Prediction Error",
    }

    for name in score_names:
        if name.startswith("abs_diff_"):
            score_dict[name] = "Absolute differenece at " + name[len("abs_diff_") :]

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
def layer_with_facets(data: pl.DataFrame, *charts: List, **encodings: Dict):
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

    layered = alt.layer(*charts, data=data)

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
    return pl.read_parquet("output/scores/tables/scores.parquet")


def load_config():
    return yaml.safe_load(open("scripts/config.yaml"))


if __name__ == "__main__":
    app()
