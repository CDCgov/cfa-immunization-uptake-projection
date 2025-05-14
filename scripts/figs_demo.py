# %% Imports
import altair as alt
import polars as pl

# %% Load data and add days-elapsed-within-season to each.
flu_natl = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_flu_natl.parquet"
    )
    .with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        ),
        obs_upper=pl.col("estimate") + 1.96 * pl.col("sem"),
        obs_lower=pl.col("estimate") - 1.96 * pl.col("sem"),
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
    .rename({"estimate": "obs"})
)
flu_state = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_flu_state.parquet"
    )
    .with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        ),
        obs_upper=pl.col("estimate") + 1.96 * pl.col("sem"),
        obs_lower=pl.col("estimate") - 1.96 * pl.col("sem"),
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
    .rename({"estimate": "obs"})
)
cov_natl = (
    pl.read_parquet(
        "/home/tec0/cfa-immunization-uptake-projection/output/data/nis_raw_cov_natl.parquet"
    )
    .with_columns(
        start=(pl.col("season").str.slice(0, 4) + pl.lit("-07-01")).str.strptime(
            pl.Date, "%Y-%m-%d"
        ),
        obs_upper=pl.col("estimate") + 1.96 * pl.col("sem"),
        obs_lower=pl.col("estimate") - 1.96 * pl.col("sem"),
    )
    .with_columns(elapsed=(pl.col("time_end") - pl.col("start")).dt.total_days())
    .drop("start")
    .rename({"estimate": "obs"})
)

# %% Plot national scale covid vax across seasons
(
    alt.Chart(cov_natl)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Observed Uptake"),
        color="season:N",
    )
).display()

# %% Plot national scale flu vax across seasons
(
    alt.Chart(flu_natl)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Observed Uptake"),
        color="season:N",
    )
).display()

# %% Plot state scale flu vax across seasons
alt.data_transformers.disable_max_rows()
(
    alt.Chart(flu_state)
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
).display()

# %% Plot uptake for one state, with empirical uncertainty
one_state = flu_state.filter(
    (pl.col("geography") == "California") & (pl.col("season") == "2023/2024")
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

# %% Posterior predictive checks
pred = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/postchecks.parquet"
).drop(["forecast_start", "forecast_end", "model"])
pred_summ = (
    pred.group_by(["time_end", "geography", "season"])
    .agg(
        estimate=pl.col("estimate").mean(),
        upper=pl.col("estimate").quantile(0.975),
        lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        flu_state.select(["time_end", "geography", "season", "estimate", "elapsed"]),
        on=["time_end", "geography", "season"],
        how="left",
    )
    .rename({"estimate_right": "obs"})
)

# %% Plot posterior prediction vs. data for one state
pred_one_state = pred_summ.filter(
    (pl.col("geography") == "Missouri") & (pl.col("season") == "2015/2016")
)
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

# %% Plot posterior predictions for all states in one season
pred_summ_one_season = pred_summ.filter(pl.col("season") == "2009/2010")
alt.data_transformers.disable_max_rows()
plot = (
    (
        (
            alt.Chart(pred_summ_one_season)
            .mark_point(color="black")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "obs:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
            )
        )
        + (
            alt.Chart(pred_summ_one_season)
            .mark_line(color="green")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "estimate:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
            )
        )
        + (
            alt.Chart(pred_summ_one_season)
            .mark_area(color="green", opacity=0.3)
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "upper:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="lower:Q",
            )
        )
    )
    .facet("geography", columns=9)
    .configure_header(labelFontSize=40)
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
        flu_state.select(["time_end", "geography", "estimate", "elapsed"]),
        on=["time_end", "geography"],
        how="left",
    )
    .rename({"estimate_right": "obs"})
)

# %% Plot prediction vs. data for 2023/2024 for one state
pred_one_state = pred_summ.filter((pl.col("geography") == "California"))
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
    .mark_area(color="tomato", opacity=0.3)
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("lower", title="Uptake"),
        y2="upper",
    )
    + alt.Chart(pred_one_state)
    .mark_line(color="tomato")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("estimate:Q", title="Uptake"),
    )
)
plot.display()

# %% Plot retrospective forecasts for all states
alt.data_transformers.disable_max_rows()
plot = (
    (
        (
            alt.Chart(pred_summ)
            .mark_point(color="black")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "obs:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
            )
        )
        + (
            alt.Chart(pred_summ)
            .mark_line(color="tomato")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "estimate:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
            )
        )
        + (
            alt.Chart(pred_summ)
            .mark_area(color="tomato", opacity=0.3)
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "upper:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="lower:Q",
            )
        )
    )
    .facet("geography", columns=9)
    .configure_header(labelFontSize=40)
)
plot.display()
