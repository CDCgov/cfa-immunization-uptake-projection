import argparse
from typing import Any, Dict, List

import altair as alt
import numpy as np
import polars as pl
import streamlit as st
import yaml


@st.cache_data
def load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


@st.cache_data
def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


@st.cache_data
def summarize_preds(
    path: str, groups_to_include: tuple, ci_level: float
) -> pl.DataFrame:
    half_alpha = (1.0 - ci_level) / 2

    return (
        pl.scan_parquet(path)
        .group_by(groups_to_include)
        .agg(
            lower=pl.col("estimate").quantile(half_alpha),
            upper=pl.col("estimate").quantile(1.0 - half_alpha),
            mean=pl.col("estimate").mean(),
        )
        .sort("time_end")
        .collect()
    )


@st.cache_data
def sample_preds(path: str, n_samples: int, seed: int = 42) -> pl.DataFrame:
    preds = pl.scan_parquet(path).with_columns(pl.col("sample_id").cast(pl.Int64))

    max_id = preds.select(pl.col("sample_id").max()).collect().item()
    min_id = preds.select(pl.col("sample_id").min()).collect().item()
    assert isinstance(min_id, int)
    assert isinstance(max_id, int)

    # draw indices of trajectories randomly #
    rng = np.random.default_rng(seed=seed)

    selected_ids = rng.integers(low=min_id, high=max_id + 1, size=n_samples)

    return preds.filter(pl.col("sample_id").is_in(selected_ids)).collect()


def app(data_path: str, config_path: str, scores_path: str, preds_path: str):
    # load data from caches
    obs = load_parquet(data_path)
    scores = load_parquet(scores_path)
    config = load_config(config_path)

    st.title("Vaccine Uptake Forecasts")

    # multiple tabs
    tab1, tab2, tab3 = st.tabs(["Trajectories", "Summary", "Evaluation"])

    with tab1:
        plot_trajectories(obs=obs, preds_path=preds_path, config=config)

    with tab2:
        plot_summary(obs=obs, preds_path=preds_path, config=config)

    with tab3:
        plot_evaluation(scores=scores, config=config)


def plot_trajectories(obs: pl.DataFrame, preds_path: str, config: Dict[str, Any]):
    """Plot the individual trajectories of the forecasts with data.

    Args:
        obs: Observed data.
        preds_path: Path to predictions data.
        config: Configuration dictionary.

    Note:
        User options to select the dimensions to group the data, including:
        column and row. Other grouping factors that haven't been selected will
        be used to filter the data.
    """

    # set up plot encodings
    encodings = {
        "x": alt.X("time_end:T", title="Observation date"),
        "y": alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        "color": alt.Color("sample_id:N", title="Trajectories"),
    }

    # select which data dimension to put into which plot channel
    st.header("Plot options")

    # how many trajectories to show?
    n_samples = st.number_input("Number of forecasts", value=3, min_value=1)

    pred = sample_preds(preds_path, n_samples=n_samples)

    st.subheader("Data channels")
    dimensions = ["model", "forecast_date"] + config["groups"]
    default_channels = {
        "column": ("Column", "forecast_date"),
        "row": ("Row", "model"),
    }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value

        index = dimensions.index(default_dim)
        dim = st.selectbox(
            f"{channel} by",
            options=dimensions,
            index=index,
            key=f"{channel}_{idx}_trajectories",
        )

        if dim != "None":
            dimensions.remove(dim)
            encodings[key] = getattr(alt, channel)(dim)

    # filter for specific values of the remaining dimensions
    st.header("Data filters")

    if missing_columns := set(dimensions) - set(pred.columns):
        raise RuntimeError(f"Missing columns: {missing_columns}")

    for dim in dimensions:
        filter_val = st.selectbox(dim, options=pred[dim].unique().sort().to_list())
        pred = pred.filter(pl.col(dim) == pl.lit(filter_val))

    # merge observed data with prediction by the combination of models and forecast starts
    model_forecast_dates = pred.select(["model", "forecast_date"]).unique()
    plot_obs = obs.join(model_forecast_dates, how="cross").filter(
        pl.col(factor).is_in(pred[factor].unique().implode())
        for factor in config["groups"]
    )

    groupings = ["model", "forecast_date", "time_end"] + config["groups"]

    data = pred.join(plot_obs, on=groupings).rename({"estimate_right": "observed"})

    obs_chart = (
        alt.Chart(data)
        .encode(
            x="time_end:T",
            y="observed:Q",
            tooltip=[
                alt.Tooltip("time_end", title="Observation date"),
                alt.Tooltip("observed", title="Observed uptake"),
            ],
        )
        .transform_calculate(type="'observed'")
        .mark_point()
        .encode(shape=alt.Shape("type:N", title="Type"))
    )

    pred_chart = (
        alt.Chart(data)
        .encode(
            x="time_end:T",
            y="estimate:Q",
            tooltip=[
                alt.Tooltip("time_end", title="Observation date"),
                alt.Tooltip("estimate", title="Predicted uptake"),
            ],
        )
        .transform_calculate(type="'predicted'")
        .mark_line()
        .encode(strokeDash=alt.StrokeDash("type:N", title=None))
    )

    chart = layer_with_facets([obs_chart, pred_chart], encodings)

    st.altair_chart(chart)


def plot_summary(obs: pl.DataFrame, preds_path: str, config: Dict[str, Any]):
    """Plot the 95% PI with mean estimate of forecasts with data.

    Args:
        obs: Observed data.
        preds_path: Path to predictions data.
        config: Configuration dictionary.

    Note:
        User options to select the dimensions to group the data, including:
        row, column, and color. Other grouping factors that haven't been
        selected will be used to filter the data.
    """
    # summarize sample predictions by grouping factors
    groups_to_include = ["model", "forecast_date", "time_end"] + config["groups"]
    pred = summarize_preds(
        path=preds_path,
        groups_to_include=tuple(groups_to_include),
        ci_level=config["plots"]["ci_level"],
    )

    encodings = {}

    # data process: merge observed data with prediction by combinations of model and forecast start #
    forecast_dates = pred.select(["model", "forecast_date"]).unique()
    plot_obs = obs.join(forecast_dates, how="cross").filter(
        pl.col(factor).is_in(pred[factor].unique().implode())
        for factor in config["groups"]
    )

    data = pred.join(plot_obs, on=groups_to_include)

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["model", "forecast_date"] + config["groups"]

    if "season" in dimensions:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "forecast_date"),
            "row": ("Row", "season"),
        }
    else:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "forecast_date"),
            "row": ("Row", "None"),
        }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        index = dimensions.index(default_dim)
        dim = st.selectbox(
            f"{channel} by",
            options=dimensions,
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
            options=pred[dim].unique().sort().to_list(),
            key=f"{dim}_{idx}_summary",
        )

        data = data.filter(pl.col(dim) == pl.lit(filter_val))

    obs_chart = (
        alt.Chart(data)
        .encode(
            x="time_end:T",
            y="estimate:Q",
            tooltip=[
                alt.Tooltip("time_end", title="Observation date"),
                alt.Tooltip("estimate", title="Observed uptake"),
            ],
        )
        .transform_calculate(type="'observed'")
        .mark_point()
        .encode(shape=alt.Shape("type:N", title="Type"))
    )

    pred_chart = (
        alt.Chart(data)
        .encode(
            x="time_end:T",
            y="mean:Q",
            tooltip=[
                alt.Tooltip("time_end", title="Observation date"),
                alt.Tooltip("mean", title="Predicted mean"),
            ],
        )
        .transform_calculate(type="'predicted mean'")
        .mark_line()
        .encode(strokeDash=alt.StrokeDash("type:N", title=None))
    )

    interval_chart = (
        alt.Chart(data)
        .mark_area(opacity=0.3)
        .encode(
            alt.X(
                "time_end:T",
                axis=alt.Axis(format="%Y-%m", tickCount="month"),
            ),
            y="lower:Q",
            y2="upper:Q",
            tooltip=[
                alt.Tooltip("time_end", title="Observation date"),
                alt.Tooltip("lower", title="Lower bound"),
                alt.Tooltip("upper", title="Upper bound"),
            ],
        )
    )

    chart_list = [interval_chart, obs_chart, pred_chart]
    chart = layer_with_facets(chart_list, encodings)

    st.altair_chart(chart)


def plot_evaluation(scores: pl.DataFrame, config: Dict[str, Any]):
    """Plot the evaluation scores over forecast start.

    Args:
        scores: Evaluation scores of the forecasts.
        config: Configuration dictionary.

    Note:
        User can select the dimensions to group the data, including: row, column,
        and color. Other grouping factors that haven't been selected will be used
        to filter the data.
    """

    encodings = {
        "x": alt.X("forecast_date:T", title="Forecast start date"),
        "y": alt.Y("score_value:Q", title="Score value"),
    }

    score_funs = scores["score_fun"].unique()

    if "mspe" in score_funs:
        score_dict = {
            "mspe": "Mean Squared Prediction Error",
            "eos_abs_diff": "End-of-season absolute error",
        }
    else:
        score_dict = {}

    for name in score_funs:
        if name.startswith("abs_diff_"):
            score_dict[name] = "Absolute difference at " + name[len("abs_diff_") :]

    # every score name should have a label for the plot
    assert set(score_funs).issubset(score_dict.keys()), (
        f"Missing score names: {set(score_funs) - set(score_dict.keys())}"
    )

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["model", "score_fun"] + config["groups"]
    if "season" in dimensions:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "score_fun"),
            "row": ("Row", "season"),
        }
    else:
        default_channels = {
            "color": ("Color", "model"),
            "column": ("Column", "score_fun"),
            "row": ("Row", "None"),
        }

    for idx, item in enumerate(default_channels.items()):
        key, value = item
        channel, default_dim = value
        index = dimensions.index(default_dim)
        dim = st.selectbox(
            f"{channel} by",
            options=dimensions,
            index=index,
            key=f"{channel}_{idx}_eval",
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

    plot_score = plot_score.with_columns(pl.col("score_fun").replace_strict(score_dict))

    chart = (
        alt.Chart(plot_score)
        .mark_point()
        .encode(**encodings)
        .resolve_scale(y="independent")
    )

    st.altair_chart(chart)


## helper: feed correct argument to altair ##
def layer_with_facets(charts: List, encodings: Dict):
    """Layer multiple Altair charts with correct faceting and encoding.

    Args:
        charts: List of alt.Chart objects to be layered.
        encodings: Encodings to be applied to the charts, including row, column,
            and other encodings.

    Returns:
        Layered and faceted Altair chart.

    Note:
        Because alt.layer.facet() only takes row and column and .encode() only
        takes color, this function makes sure correct arguments fall into correct command.
    """

    row_enc = encodings["row"]
    col_enc = encodings["column"]
    other_encs = {k: v for k, v in encodings.items() if k not in ["row", "column"]}

    layered = alt.layer(*charts).interactive()

    layered = layered.encode(**other_encs)

    return layered.facet(row=row_enc, column=col_enc)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="observed data", required=True)
    p.add_argument("--preds", help="predictions", required=True)
    p.add_argument("--scores", help="score metrics", required=True)
    p.add_argument("--config", help="config yaml file", required=True)
    args = p.parse_args()

    app(
        config_path=args.config,
        scores_path=args.scores,
        data_path=args.data,
        preds_path=args.preds,
    )
