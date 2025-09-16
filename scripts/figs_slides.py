# %% Imports
import altair as alt
import numpy as np
import polars as pl


# %% Define some functions
def load_data(path):
    return (
        pl.read_parquet(path)
        .with_columns(
            start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
                pl.Date, "%Y-%m-%d"
            ),
            obs_upper=pl.col("estimate") + 1.96 * pl.col("sem"),
            obs_lower=pl.col("estimate") - 1.96 * pl.col("sem"),
        )
        .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
        .drop(["start", "sem", "N_vax", "N_tot"])
        .rename({"estimate": "obs"})
    )


def load_pred(path, data):
    pred = pl.read_parquet(path).drop(["forecast_start", "forecast_end", "model"])
    return (
        pred.group_by(["time_end", "geography", "season"])
        .agg(
            est=pl.col("estimate").mean(),
            est_upper=pl.col("estimate").quantile(0.975),
            est_lower=pl.col("estimate").quantile(0.025),
        )
        .join(
            data,
            on=["time_end", "geography", "season"],
            how="left",
        )
    )


def plot_uptake(data, color="green", season_start_month=7, upper_bound=0.6):
    # Set the same year for every date so that seasons can be plotted
    # overlapping instead of sequential.
    # Oddly, the second 'with_columns' doesn't work because 1999-2-29
    # is not a date, even though the conditional should prevent this
    # date from ever being made, hence the hacky replacement of the 29th
    # with the 28th whenever it appears.
    data = data.with_columns(
        time_end=pl.when(pl.col("time_end").dt.day() == 29)
        .then(pl.col("time_end").dt.replace(day=28))
        .otherwise(pl.col("time_end"))
    ).with_columns(
        time_end=pl.when(pl.col("time_end").dt.month() < season_start_month)
        .then(pl.col("time_end").dt.replace(year=2000))
        .otherwise(pl.col("time_end").dt.replace(year=1999))
    )
    plot_list = []
    if "season" in data.columns:
        plot_list.append(
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X(
                    "time_end:T",
                    title="Month",
                    axis=alt.Axis(format="%b", labelAngle=45),
                ),
                y=alt.Y(
                    "obs:Q", title="Uptake", scale=alt.Scale(domain=[0, upper_bound])
                ),
                color="season:N",
            )
        )
    else:
        if "obs" in data.columns:
            plot_list.append(
                alt.Chart(data)
                .mark_errorbar(color="black")
                .encode(
                    x=alt.X(
                        "time_end:T",
                        title="Month",
                        axis=alt.Axis(format="%b", labelAngle=45),
                    ),
                    y=alt.Y(
                        "obs_lower",
                        title="Uptake",
                        scale=alt.Scale(domain=[0, upper_bound]),
                    ),
                    y2="obs_upper",
                )
                + alt.Chart(data)
                .mark_point(color="black")
                .encode(
                    x=alt.X(
                        "time_end:T",
                        title="Month",
                        axis=alt.Axis(format="%b", labelAngle=45),
                    ),
                    y=alt.Y(
                        "obs:Q",
                        title="Uptake",
                        scale=alt.Scale(domain=[0, upper_bound]),
                    ),
                )
            )
        if "est" in data.columns:
            plot_list.append(
                alt.Chart(data)
                .mark_area(color=color, opacity=0.3)
                .encode(
                    x=alt.X(
                        "time_end:T",
                        title="Month",
                        axis=alt.Axis(format="%b", labelAngle=45),
                    ),
                    y=alt.Y(
                        "est_lower",
                        title="Uptake",
                        scale=alt.Scale(domain=[0, upper_bound]),
                    ),
                    y2="est_upper",
                )
                + alt.Chart(data)
                .mark_line(color=color)
                .encode(
                    x=alt.X(
                        "time_end:T",
                        title="Month",
                        axis=alt.Axis(format="%b", labelAngle=45),
                    ),
                    y=alt.Y(
                        "est:Q",
                        title="Uptake",
                        scale=alt.Scale(domain=[0, upper_bound]),
                    ),
                )
            )

    if "geography" in data.columns:
        alt.layer(*plot_list).facet("geography", columns=9).configure_header(
            labelFontSize=40
        ).configure_axis(labelFontSize=30, titleFontSize=30).display()
    elif "forecast_start" in data.columns:
        alt.layer(*plot_list).facet("forecast_start", columns=4).configure_header(
            labelFontSize=40
        ).configure_axis(labelFontSize=30, titleFontSize=30).display()
    else:
        alt.layer(*plot_list).display()


# %% Load data and add days-elapsed-within-season to each.
flu_natl = load_data(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_flu_natl.parquet"
)
flu_state = load_data(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_flu_state.parquet"
)
cov_natl = load_data(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_cov_natl.parquet"
)

# %% Plot national scale covid vax across seasons
plot_uptake(cov_natl, season_start_month=9)

# %% Plot national scale flu vax across seasons
plot_uptake(flu_natl, season_start_month=7)

# %% Plot state scale flu vax across seasons
alt.data_transformers.disable_max_rows()
plot_uptake(flu_state)

# %% Plot uptake for one state in one season, with empirical uncertainty
flu_state_sub = flu_state.filter(
    (pl.col("geography") == "Maryland") & (pl.col("season") == "2020/2021")
).drop(["geography", "season"])
plot_uptake(flu_state_sub)

# %% Load posterior checks and forecasts
postcheck = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/test/postchecks.parquet",
    flu_state,
)
forecast = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/test/forecasts.parquet",
    flu_state,
).drop("season")

# %% Plot posterior prediction vs. data for one state
postcheck_sub = postcheck.filter(
    (pl.col("geography") == "Maryland") & (pl.col("season") == "2020/2021")
).drop(["geography", "season"])
plot_uptake(postcheck_sub)

# %% Plot posterior predictions for all states in one season
postcheck_sub = (
    postcheck.filter(pl.col("season") == "2009/2010")
    .drop("season")
    .with_columns(
        est=pl.when(pl.col("obs").is_null()).then(None).otherwise(pl.col("est")),
        est_upper=pl.when(pl.col("obs").is_null())
        .then(None)
        .otherwise(pl.col("est_upper")),
        est_lower=pl.when(pl.col("obs").is_null())
        .then(None)
        .otherwise(pl.col("est_lower")),
    )
)
alt.data_transformers.disable_max_rows()
plot_uptake(postcheck_sub)

# %% Plot prediction vs. data for 2023/2024 for one state
forecast_sub = forecast.filter((pl.col("geography") == "Pennsylvania")).drop(
    "geography"
)
plot_uptake(forecast_sub, "tomato")

# %% Plot retrospective forecasts for all states
alt.data_transformers.disable_max_rows()
plot_uptake(forecast, "tomato")

# %% Plot final uptake correlation across all postchecks
postcheck_last = (
    postcheck.drop_nulls()
    .filter(~pl.col("season").is_in(["2023/2024"]))
    .sort("elapsed", descending=True)
    .group_by(["geography", "season"])
    .agg(pl.col("*").first())
)
alt.Chart(postcheck_last).mark_point(color="green").encode(
    x=alt.X("obs:Q"), y=alt.Y("est:Q")
)
x = postcheck_last["obs"].to_numpy()
y = postcheck_last["est"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% Plot final uptake correlation for postchecks in just one year
postcheck_last_sub = postcheck_last.filter(pl.col("season") == "2022/2023")
alt.Chart(pl.DataFrame({"x": [0.0, 0.55], "y": [0.0, 0.55]})).mark_line(
    color="black", strokeDash=[5, 5]
).encode(x="x", y="y") + alt.Chart(postcheck_last_sub).mark_point(color="green").encode(
    x=alt.X("obs:Q", title="Observed May 31 Uptake"),
    y=alt.Y("est:Q", title="Predicted May 31 Uptake"),
)
x = postcheck_last_sub["obs"].to_numpy()
y = postcheck_last_sub["est"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% Plot final uptake correlation for forecasts
forecast_last = (
    forecast.drop_nulls()
    .sort("elapsed", descending=True)
    .group_by(["geography"])
    .agg(pl.col("*").first())
)
alt.Chart(pl.DataFrame({"x": [0.0, 0.55], "y": [0.0, 0.55]})).mark_line(
    color="black", strokeDash=[5, 5]
).encode(x="x", y="y") + alt.Chart(forecast_last).mark_point(color="tomato").encode(
    x=alt.X("obs:Q", title="Observed May 31 Uptake"),
    y=alt.Y("est:Q", title="Predicted May 31 Uptake"),
)
x = forecast_last["obs"].to_numpy()
y = forecast_last["est"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)


# %% Load posterior checks and forecasts for flu nationally
postcheck = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/usa/postchecks.parquet",
    )
    .drop("model")
    .group_by(["time_end", "season", "forecast_start"])
    .agg(
        est=pl.col("estimate").mean(),
        est_upper=pl.col("estimate").quantile(0.975),
        est_lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        flu_natl,
        on=["time_end", "season"],
        how="left",
    )
)

forecast = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/usa/forecasts.parquet",
    )
    .drop("model")
    .group_by(["time_end", "season", "forecast_start"])
    .agg(
        est=pl.col("estimate").mean(),
        est_upper=pl.col("estimate").quantile(0.975),
        est_lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        flu_natl,
        on=["time_end", "season"],
        how="left",
    )
).drop("season")

# %% Plot national forecasts for 2023/2024 across forecast dates
plot_uptake(forecast, color="tomato")

# %% Load scores for national forecasts of 2023/2024
scores = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/scores/usa/scores.parquet"
    )
    .with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        )
    )
    .with_columns(elapsed=(pl.col("forecast_start") - pl.col("start")).dt.total_days())
    .drop(["model", "start"])
)

# %% Plot MSPE by forecast date for national forecast
alt.Chart(scores.filter(pl.col("score_name") == "mspe")).mark_point(
    color="black"
).encode(
    x=alt.X(
        "forecast_start:T",
        title="Month",
        axis=alt.Axis(format="%b", labelAngle=45),
    ),
    y=alt.Y("score_value:Q", title="Mean Squared Prediction Error"),
)

# %% Plot Abs Diff by forecast date for national forecast
alt.Chart(scores.filter(pl.col("score_name") == "abs_diff_2024-05-31")).mark_point(
    color="black"
).encode(
    x=alt.X(
        "forecast_start:T",
        title="Month",
        axis=alt.Axis(format="%b", labelAngle=45),
    ),
    y=alt.Y("score_value:Q", title="Absolute Error on May 31"),
)
