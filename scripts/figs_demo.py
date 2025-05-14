# %% Imports
import altair as alt
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


def plot_uptake(data, color="green"):
    plot_list = []
    if "season" in data.columns:
        plot_list.append(
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X("elapsed:Q", title="Days since July 1"),
                y=alt.Y("obs:Q", title="Uptake"),
                color="season:N",
            )
        )
    else:
        if "obs" in data.columns:
            plot_list.append(
                alt.Chart(data)
                .mark_errorbar(color="black")
                .encode(
                    x=alt.X("elapsed:Q", title="Days since July 1"),
                    y=alt.Y("obs_lower", title="Uptake"),
                    y2="obs_upper",
                )
                + alt.Chart(data)
                .mark_point(color="black")
                .encode(
                    x=alt.X("elapsed:Q", title="Days since July 1"),
                    y=alt.Y("obs:Q", title="Uptake"),
                )
            )
        if "est" in data.columns:
            plot_list.append(
                alt.Chart(data)
                .mark_area(color=color, opacity=0.3)
                .encode(
                    x=alt.X("elapsed:Q", title="Days since July 1"),
                    y=alt.Y("est_lower", title="Uptake"),
                    y2="est_upper",
                )
                + alt.Chart(data)
                .mark_line(color=color)
                .encode(
                    x=alt.X("elapsed:Q", title="Days since July 1"),
                    y=alt.Y("est:Q", title="Uptake"),
                )
            )

    if "geography" in data.columns:
        alt.layer(*plot_list).facet("geography", columns=9).configure_header(
            labelFontSize=40
        ).display()
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
plot_uptake(cov_natl)

# %% Plot national scale flu vax across seasons
plot_uptake(flu_natl)

# %% Plot state scale flu vax across seasons
alt.data_transformers.disable_max_rows()
plot_uptake(flu_state)

# %% Plot uptake for one state in one season, with empirical uncertainty
flu_state_sub = flu_state.filter(
    (pl.col("geography") == "California") & (pl.col("season") == "2023/2024")
).drop(["geography", "season"])
plot_uptake(flu_state_sub)

# %% Load posterior checks and forecasts
postcheck = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/postchecks.parquet",
    flu_state,
)
forecast = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/tables/forecasts.parquet",
    flu_state,
).drop("season")

# %% Plot posterior prediction vs. data for one state
postcheck_sub = postcheck.filter(
    (pl.col("geography") == "Missouri") & (pl.col("season") == "2015/2016")
).drop(["geography", "season"])
plot_uptake(postcheck_sub)

# %% Plot posterior predictions for all states in one season
postcheck_sub = postcheck.filter(pl.col("season") == "2009/2010").drop("season")
alt.data_transformers.disable_max_rows()
plot_uptake(postcheck_sub)

# %% Plot prediction vs. data for 2023/2024 for one state
forecast_sub = forecast.filter((pl.col("geography") == "Wyoming")).drop("geography")
plot_uptake(forecast_sub, "tomato")

# %% Plot retrospective forecasts for all states
alt.data_transformers.disable_max_rows()
plot_uptake(forecast, "tomato")
