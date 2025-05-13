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
alt.Chart(covid).mark_line().encode(
    x=alt.X("elapsed:Q", title="Days since July 1"),
    y=alt.Y("estimate:Q", title="Observed Uptake"),
    color="season:N",
)

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
alt.Chart(natl).mark_line().encode(
    x=alt.X("elapsed:Q", title="Days since July 1"),
    y=alt.Y("estimate:Q", title="Observed Uptake"),
    color="season:N",
)

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
alt.Chart(state).mark_line().encode(
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
# %% Plot uptake for one state, with uncertainty
one_state = state.filter(
    (pl.col("geography") == "Kansas") & (pl.col("season") == "2020/2021")
).with_columns(
    estimate_hi=pl.col("estimate") + 2 * pl.col("sem"),
    estimate_lo=pl.col("estimate") - 2 * pl.col("sem"),
)
alt.Chart(one_state).mark_area(color="black", opacity=0.3).encode(
    x=alt.X("elapsed:Q", title="Days since July 1"), y="estimate_lo", y2="estimate_hi"
) + alt.Chart(one_state).mark_line(color="black").encode(
    x=alt.X("elapsed:Q", title="Days since July 1"),
    y=alt.Y("estimate:Q", title="Observed Uptake"),
)
