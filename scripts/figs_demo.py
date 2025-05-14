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
    .drop(["start", "sem", "N_vax", "N_tot"])
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
    .drop(["start", "sem", "N_vax", "N_tot"])
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
    .drop(["start", "sem", "N_vax", "N_tot"])
    .rename({"estimate": "obs"})
)

# %% Plot national scale covid vax across seasons
(
    alt.Chart(cov_natl)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs:Q", title="Observed Uptake"),
        color="season:N",
    )
).display()

# %% Plot national scale flu vax across seasons
(
    alt.Chart(flu_natl)
    .mark_line()
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs:Q", title="Observed Uptake"),
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
            "obs:Q",
            title="Observed Uptake",
            axis=alt.Axis(labelFontSize=20, titleFontSize=30),
        ),
        color="season:N",
        facet=alt.Facet("geography", columns=9, header=alt.Header(labelFontSize=40)),
    )
).display()

# %% Plot uptake for one state in one season, with empirical uncertainty
flu_state_sub = flu_state.filter(
    (pl.col("geography") == "California") & (pl.col("season") == "2023/2024")
)
(
    alt.Chart(flu_state_sub)
    .mark_errorbar(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs_lower", title="Observed Uptake"),
        y2="obs_upper",
    )
    + alt.Chart(flu_state_sub)
    .mark_point(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs:Q", title="Observed Uptake"),
    )
).display()

# %% Posterior predictive checks
post_check = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/postchecks.parquet"
).drop(["forecast_start", "forecast_end", "model"])
post_check_summ = (
    post_check.group_by(["time_end", "geography", "season"])
    .agg(
        est=pl.col("estimate").mean(),
        est_upper=pl.col("estimate").quantile(0.975),
        est_lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        flu_state,
        on=["time_end", "geography", "season"],
        how="left",
    )
)

# %% Plot posterior prediction vs. data for one state
post_check_summ_sub = post_check_summ.filter(
    (pl.col("geography") == "Missouri") & (pl.col("season") == "2015/2016")
)
(
    alt.Chart(post_check_summ_sub)
    .mark_errorbar(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs_lower", title="Uptake"),
        y2="obs_upper",
    )
    + alt.Chart(post_check_summ_sub)
    .mark_point(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs:Q", title="Uptake"),
    )
    + alt.Chart(post_check_summ_sub)
    .mark_area(color="green", opacity=0.3)
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("est_lower", title="Uptake"),
        y2="est_upper",
    )
    + alt.Chart(post_check_summ_sub)
    .mark_line(color="green")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("est:Q", title="Uptake"),
    )
).display()

# %% Plot posterior predictions for all states in one season
post_check_summ_sub = post_check.filter(pl.col("season") == "2009/2010")
alt.data_transformers.disable_max_rows()
(
    (
        (
            alt.Chart(post_check_summ_sub)
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
            alt.Chart(post_check_summ_sub)
            .mark_errorbar(color="black")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "obs_upper:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="obs_lower:Q",
            )
        )
        + (
            alt.Chart(post_check_summ_sub)
            .mark_line(color="green")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "est:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
            )
        )
        + (
            alt.Chart(post_check_summ_sub)
            .mark_area(color="green", opacity=0.3)
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "est_upper:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="est_lower:Q",
            )
        )
    )
    .facet("geography", columns=9)
    .configure_header(labelFontSize=40)
).display()

# %% Load forecasts for flu by state in 2023/2024, and summarize
pred = pl.read_parquet(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/tables/forecasts.parquet"
)
pred_summ = (
    pred.group_by(["time_end", "geography"])
    .agg(
        est=pl.col("estimate").mean(),
        est_upper=pl.col("estimate").quantile(0.975),
        est_lower=pl.col("estimate").quantile(0.025),
    )
    .join(
        flu_state,
        on=["time_end", "geography", "season"],
        how="left",
    )
)

# %% Plot prediction vs. data for 2023/2024 for one state
pred_summ_sub = pred_summ.filter((pl.col("geography") == "California"))
(
    alt.Chart(pred_summ_sub)
    .mark_errorbar(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs_lower", title="Uptake"),
        y2="obs_upper",
    )
    + alt.Chart(pred_summ_sub)
    .mark_point(color="black")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("obs:Q", title="Uptake"),
    )
    + alt.Chart(pred_summ_sub)
    .mark_area(color="tomato", opacity=0.3)
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("est_lower", title="Uptake"),
        y2="est_upper",
    )
    + alt.Chart(pred_summ_sub)
    .mark_line(color="tomato")
    .encode(
        x=alt.X("elapsed:Q", title="Days since July 1"),
        y=alt.Y("est:Q", title="Uptake"),
    )
).display()

# %% Plot retrospective forecasts for all states
alt.data_transformers.disable_max_rows()
(
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
            .mark_errorbar(color="black")
            .encode(
                x=alt.X(
                    "elapsed:Q",
                    title="Days since July 1",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y=alt.Y(
                    "obs_lower:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="obs_upper:Q",
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
                    "est:Q",
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
                    "est_upper:Q",
                    title="Uptake",
                    axis=alt.Axis(labelFontSize=20, titleFontSize=30),
                ),
                y2="est_lower:Q",
            )
        )
    )
    .facet("geography", columns=9)
    .configure_header(labelFontSize=40)
).display()
