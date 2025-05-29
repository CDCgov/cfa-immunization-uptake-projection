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
postcheck_sub = postcheck.filter(pl.col("season") == "2022/2023").drop("season")
alt.data_transformers.disable_max_rows()
plot_uptake(postcheck_sub)

# %% Plot prediction vs. data for 2023/2024 for one state
forecast_sub = forecast.filter((pl.col("geography") == "Wyoming")).drop("geography")
plot_uptake(forecast_sub, "tomato")

# %% Plot retrospective forecasts for all states
alt.data_transformers.disable_max_rows()
plot_uptake(forecast, "tomato")

# %% SCRATCH WORK
# How consistent are states from season to season?
# How consistent are seasons from state to state?
# Examine the last date of each season
flu_state_last = (
    flu_state.sort("elapsed", descending=True)
    .group_by(["geography", "season"])
    .agg(pl.col("*").first())
)
alt.Chart(flu_state_last).mark_boxplot().encode(
    x=alt.X("geography:O"), y=alt.Y("obs:Q")
)
alt.Chart(flu_state_last).mark_boxplot().encode(x=alt.X("season:O"), y=alt.Y("obs:Q"))

# %% How well do pred vs. obs final uptakes correlate in postchecks?
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


# %% How well do pred vs. obs final uptakes correlate in postchecks for 2022/23 only?
postcheck_last_sub = postcheck_last.filter(pl.col("season") == "2022/2023")
alt.Chart(postcheck_last_sub).mark_point(color="green").encode(
    x=alt.X("obs:Q"), y=alt.Y("est:Q")
)
x = postcheck_last_sub["obs"].to_numpy()
y = postcheck_last_sub["est"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% How well do pred vs. obs final uptakes correlate in forecasts for 2022/23 only?
forecast_last = (
    forecast.drop_nulls()
    .sort("elapsed", descending=True)
    .group_by(["geography"])
    .agg(pl.col("*").first())
)
alt.Chart(forecast_last).mark_point(color="tomato").encode(
    x=alt.X("obs:Q"), y=alt.Y("est:Q")
)
x = forecast_last["obs"].to_numpy()
y = forecast_last["est"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% Do mistakes in 2022/2023 prediction predict mistakes in 2023/2024 prediction?
error_comparison = (
    postcheck_last_sub.with_columns(postcheck_error=pl.col("est") - pl.col("obs"))
    .select(["postcheck_error", "geography"])
    .join(
        forecast_last.with_columns(forecast_error=pl.col("est") - pl.col("obs")).select(
            ["forecast_error", "geography"]
        ),
        on="geography",
        how="left",
    )
)
training_dates = (
    postcheck.drop_nulls()
    .filter(pl.col("season").is_in(["2023/2024"]))
    .select(pl.col("geography").value_counts())
    .unnest("geography")
)
error_comparison = (
    error_comparison.join(training_dates, on="geography", how="left")
    .fill_null(0)
    .with_columns(count=pl.col("count") > 1)
)
alt.Chart(error_comparison).mark_point().encode(
    x=alt.X("postcheck_error:Q"),
    y=alt.Y("forecast_error:Q"),
    color=alt.Color("count", title=">1 datum?"),
)

x = error_comparison["postcheck_error"].to_numpy()
y = error_comparison["forecast_error"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% Was 2023/2024 just a weird year?
all_last = pl.concat(
    [
        postcheck_last,
        forecast_last.with_columns(season=pl.lit("2023/2024")).select(
            postcheck_last.columns
        ),
    ],
    how="vertical",
)
alt.Chart(all_last).mark_line(size=5).encode(x=alt.X("time_end"), y=alt.Y("obs")).facet(
    "geography", columns=9
).configure_header(labelFontSize=40)


# %% Prepare model coefficients
model_coefs = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/diagnostics/tables/model=LPLModel_forecast_start=2023-09-01_print_posterior_dist.parquet"
)
coefs = (
    model_coefs.mean()
    .transpose(include_header=True)
    .rename({"column": "coef", "column_0": "value"})
)

A = coefs.filter(pl.col("coef") == "A")["value"]
A_sigs_season = coefs.filter(pl.col("coef") == "A_sigs_season")["value"]
A_sigs_geography = coefs.filter(pl.col("coef") == "A_sigs_geography")["value"]
M = coefs.filter(pl.col("coef") == "M")["value"]
M_sigs_season = coefs.filter(pl.col("coef") == "M_sigs_season")["value"]
M_sigs_geography = coefs.filter(pl.col("coef") == "M_sigs_geography")["value"]

A_season_coefs = (
    coefs.filter(pl.col("coef").str.contains("A_devs_2"))
    .with_columns(
        A_season=pl.col("value") * A_sigs_season,
        season=pl.col("coef").str.replace("A_devs_", ""),
    )
    .drop("value", "coef")
)
A_geography_coefs = (
    coefs.filter(
        pl.col("coef").str.contains("A_devs_") & ~pl.col("coef").str.contains(r"\d")
    )
    .with_columns(
        A_geography=pl.col("value") * A_sigs_geography,
        geography=pl.col("coef").str.replace("A_devs_", "").str.replace_all("_", " "),
    )
    .drop("value", "coef")
)
M_season_coefs = (
    coefs.filter(pl.col("coef").str.contains("M_devs_2"))
    .with_columns(
        M_season=pl.col("value") * M_sigs_season,
        season=pl.col("coef").str.replace("M_devs_", ""),
    )
    .drop("value", "coef")
)
M_geography_coefs = (
    coefs.filter(
        pl.col("coef").str.contains("M_devs_") & ~pl.col("coef").str.contains(r"\d")
    )
    .with_columns(
        M_geography=pl.col("value") * M_sigs_geography,
        geography=pl.col("coef").str.replace("M_devs_", "").str.replace_all("_", " "),
    )
    .drop("value", "coef")
)

# %% How do A season deviations compare to avg seasonal final uptake
A_by_season = (
    (
        all_last.select(["geography", "season", "obs"])
        .group_by("season")
        .agg(obs_mean=pl.col("obs").mean())
    )
    .join(A_season_coefs, on="season", how="left")
    .with_columns(last_season=(pl.col("season") == "2023/2024"))
)

alt.Chart(A_by_season).mark_point().encode(
    x=alt.X(
        "obs_mean:Q",
        scale=alt.Scale(zero=False),
        title="Avg May 31 Uptake across Seasons",
    ),
    y=alt.Y("A_season:Q", title="Posterior Mean for Deviation from Avg A"),
    color=alt.Color("last_season", title="2023/2024?"),
)

x = A_by_season["obs_mean"].to_numpy()
y = A_by_season["A_season"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)


# %% How do M season deviations compare to avg seasonal final uptake
M_by_season = (
    (
        all_last.select(["geography", "season", "obs"])
        .group_by("season")
        .agg(obs_mean=pl.col("obs").mean())
    )
    .join(M_season_coefs, on="season", how="left")
    .with_columns(last_season=(pl.col("season") == "2023/2024"))
)

alt.Chart(M_by_season).mark_point().encode(
    x=alt.X(
        "obs_mean:Q",
        scale=alt.Scale(zero=False),
        title="Avg May 31 Uptake across Seasons",
    ),
    y=alt.Y("M_season:Q", title="Posterior Mean for Deviation from Avg M"),
    color=alt.Color("last_season", title="2023/2024?"),
)

x = M_by_season["obs_mean"].to_numpy()
y = M_by_season["M_season"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)


# %% How do A geographic deviations compare to avg seasonal final uptake
A_by_geography = (
    all_last.select(["geography", "season", "obs"])
    .group_by("geography")
    .agg(obs_mean=pl.col("obs").mean())
).join(A_geography_coefs, on="geography", how="left")

alt.Chart(A_by_geography).mark_point().encode(
    x=alt.X(
        "obs_mean:Q",
        scale=alt.Scale(zero=False),
        title="Avg May 31 Uptake across States",
    ),
    y=alt.Y("A_geography:Q", title="Posterior Mean for Deviation from Avg A"),
)

x = A_by_geography["obs_mean"].to_numpy()
y = A_by_geography["A_geography"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)


# %% How do M geographic deviations compare to avg seasonal final uptake
M_by_geography = (
    all_last.select(["geography", "season", "obs"])
    .group_by("geography")
    .agg(obs_mean=pl.col("obs").mean())
).join(M_geography_coefs, on="geography", how="left")

alt.Chart(M_by_geography).mark_point().encode(
    x=alt.X(
        "obs_mean:Q",
        scale=alt.Scale(zero=False),
        title="Avg May 31 Uptake across States",
    ),
    y=alt.Y("M_geography:Q", title="Posterior Mean for Deviation from Avg M"),
)

x = M_by_geography["obs_mean"].to_numpy()
y = M_by_geography["M_geography"].to_numpy()
print(np.corrcoef(x, y)[0, 1] ** 2)

# %% Do states always rank the same on May 31?
ranked_states = pl.DataFrame()
for season in postcheck_last["season"].unique().sort():
    new_column = postcheck_last.filter(
        (pl.col("season") == season)
        & (~pl.col("geography").is_in(["Puerto Rico", "Guam", "U.S. Virgin Islands"]))
    ).sort(pl.col("est"))["geography"]

    if new_column.len() == 51:
        ranked_states = ranked_states.with_columns(new_column.alias(season))

new_column = forecast_last.filter(
    ~pl.col("geography").is_in(["Puerto Rico", "Guam", "U.S. Virgin Islands"])
).sort(pl.col("est"))["geography"]
ranked_states = ranked_states.with_columns(new_column.alias("2023/2024"))

# %%
