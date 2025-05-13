# %% Imports
import altair as alt
import polars as pl

# %% Load data
natl = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_natl_flu.parquet"
)
state = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_flu.parquet"
)
covid = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw.parquet"
)

# %% Plot national scale covid immunization
covid = (
    covid.with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        )
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
)
plot = (
    alt.Chart(covid)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Observed Uptake"),
        color="season:N",
    )
)
plot.display()

# %% Plot national scale flu immunization
natl = (
    natl.with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        )
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
)
plot = (
    alt.Chart(natl)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Observed Uptake"),
        color="season:N",
    )
)
plot.display()

# %% Plot state scale flu immunization
state = (
    state.with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        )
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
)
alt.data_transformers.disable_max_rows()
plot = (
    alt.Chart(state)
    .mark_line()
    .encode(
        x=alt.X(
            "elapsed:Q",
            title="Days since July 1",
            axis=alt.Axis(labelFontSize=20, titleFontSize=30),
        ),
        y=alt.Y(
            "estimate:Q",
            title="Observed Uptake",
            axis=alt.Axis(labelFontSize=20, titleFontSize=30),
        ),
        color="season:N",
        facet=alt.Facet("geography", columns=9, header=alt.Header(labelFontSize=40)),
    )
)
plot.display()

# %% Plot uptake for one state, with uncertainty
one_state = state.filter(
    (pl.col("geography") == "Kansas") & (pl.col("season") == "2023/2024")
).with_columns(
    estimate_hi=pl.col("estimate") + 2 * pl.col("sem"),
    estimate_lo=pl.col("estimate") - 2 * pl.col("sem"),
)
plot = alt.Chart(one_state).mark_errorbar(color="black").encode(
    x=alt.X("elapsed:Q", title="Days since July 1"),
    y=alt.Y("estimate_lo", title="Observed Uptake"),
    y2="estimate_hi",
) + alt.Chart(one_state).mark_point(color="black").encode(
    x=alt.X("elapsed:Q", title="Days since July 1"),
    y=alt.Y("estimate:Q", title="Observed Uptake"),
)
plot.display()

# %% Load forecasts for flu by state in 2023/2024, and summarize
pred = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/tables/forecasts.parquet"
)
pred_summ = (
    pred.group_by(["time_end", "geography"])
    .agg(
        estimate=pl.col("estimate").mean(),
        upper=pl.col("estimate").quantile(0.975),
        lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        state.select(["time_end", "geography", "estimate", "elapsed"]),
        on=["time_end", "geography"],
        how="left",
    )
    .rename({"estimate_right": "obs"})
)

# %% Plot prediction vs. data for 2023/2024
pred_one_state = pred_summ.filter((pl.col("geography") == "Kansas"))
plot = (
    alt.Chart(one_state)
    .mark_errorbar(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate_lo", title="Uptake"),
        y2="estimate_hi",
    )
    + alt.Chart(one_state)
    .mark_point(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Uptake"),
    )
    + alt.Chart(pred_one_state)
    .mark_area(color="green", opacity=0.3)
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("lower", title="Uptake"),
        y2="upper",
    )
    + alt.Chart(pred_one_state)
    .mark_line(color="green")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Uptake"),
    )
)
plot.display()
