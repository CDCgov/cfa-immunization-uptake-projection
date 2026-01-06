# %% Imports
import altair as alt
import polars as pl

# %% US state abbreviations
state_to_abbrv = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "Guam": "GU",
    "Puerto Rico": "PR",
    "U.S. Virgin Islands": "VI",
}


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
    pred = pl.read_parquet(path).drop(["forecast_date", "forecast_end", "model"])
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


season_colors = [
    "#1f77b4",
    "#aec7e8",  # Blues
    "#ff7f0e",
    "#ffbb78",  # Oranges
    "#2ca02c",
    "#98df8a",  # Greens
    "#d62728",
    "#ff9896",  # Reds
    "#9467bd",
    "#c5b0d5",  # Purples
    "#8c564b",
    "#c49c94",  # Browns
    "#e377c2",
    "#f7b6d2",  # Pinks
    "#17becf",
    "#9edae5",  # Teals
    "#bcbd22",
    "#dbdb8d",  # Yellow-Greens
    "#7f7f7f",
    "#c7c7c7",  # Grays
]


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
                color=alt.Color("season:N", scale=alt.Scale(range=season_colors)),
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
    elif "forecast_date" in data.columns:
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

# %% Load posterior checks and forecasts
postcheck = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/test/postchecks.parquet",
    flu_state,
)
forecast = load_pred(
    "/home/tec0/cfa-immunization-uptake-projection/output/forecasts/test/forecasts.parquet",
    flu_state,
)
pred = pl.concat(
    [
        postcheck.with_columns(type=pl.lit("postcheck")),
        forecast.with_columns(
            type=pl.lit("forecast"),
        ),
    ]
)

# %% FIGURE 1A: National flu vaccine uptake by season
plot_uptake(flu_natl, season_start_month=7)

# %% FIGURE 1B: State flu vaccine uptake in 2015/2016
plot_data = (
    flu_state.filter(pl.col("season") == "2015/2016")
    .with_columns(
        time_end=pl.when(pl.col("time_end").dt.day() == 29)
        .then(pl.col("time_end").dt.replace(day=28))
        .otherwise(pl.col("time_end"))
    )
    .with_columns(
        time_end=pl.when(pl.col("time_end").dt.month() < 7)
        .then(pl.col("time_end").dt.replace(year=2000))
        .otherwise(pl.col("time_end").dt.replace(year=1999))
    )
    .rename({"geography": "state"})
)
alt.Chart(plot_data).mark_line(color="gray", opacity=0.25).encode(
    x=alt.X(
        "time_end:T",
        title="Month",
        axis=alt.Axis(format="%b", labelAngle=45),
    ),
    y=alt.Y("obs:Q", title="Uptake", scale=alt.Scale(domain=[0, 0.65])),
    color=alt.Color("state:N", scale=alt.Scale(range=["gray"]), legend=None),
)

# %% Prepare May 31 metrics for Figure 2
may31 = flu_state.filter(pl.col("time_end").dt.month() == 5).with_columns(
    state=pl.col("geography").replace(state_to_abbrv),
)
may31_mean_state = (
    may31.group_by("geography")
    .agg(avg=pl.col("obs").mean())
    .with_columns(
        state=pl.col("geography").replace(state_to_abbrv),
    )
)
may31_mean_season = may31.group_by("season").agg(avg=pl.col("obs").mean())

# %% FIGURE 1C: Distribution of May 31 uptake across states stratified by season
alt.Chart(may31).mark_point(filled=True, size=75, opacity=0.5).encode(
    y=alt.Y(
        "obs:Q",
        title="May 31 Uptake",
        axis=alt.Axis(titleFontSize=12, labelFontSize=10),
    ),
    x=alt.X(
        "season:N",
        title="Season",
        axis=alt.Axis(titleFontSize=12, labelFontSize=10, labelAngle=45),
    ),
    color=alt.Color("season:N", scale=alt.Scale(range=season_colors)),
) + alt.Chart(may31_mean_season).mark_point(
    color="black", shape="square", size=100, opacity=1.0
).encode(
    y=alt.Y("avg:Q", title="May 31 Uptake"),
    x=alt.X("season:N", title="Season"),
)

# %% FIGURE 1D: Distribution of May 31 uptake across seasons stratified by state
alt.Chart(may31).mark_point(filled=True, size=75, opacity=0.75).encode(
    y=alt.Y(
        "obs:Q",
        title="May 31 Uptake",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16),
    ),
    x=alt.X(
        "state:N",
        title="State",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
    ),
    color=alt.Color("season:N", scale=alt.Scale(range=season_colors)),
) + alt.Chart(may31_mean_state).mark_point(
    color="black", shape="square", size=100, opacity=1.0
).encode(
    y=alt.Y("avg:Q", title="May 31 Uptake"),
    x=alt.X("state:N", title="State"),
)

# %% Prepare MSPE metrics for Figure 2
mspe = (
    pred.drop_nulls()
    .with_columns(sqerr=(pl.col("est") - pl.col("obs")) ** 2)
    .group_by(["geography", "season", "type"])
    .agg(mspe=pl.col("sqerr").mean())
    .with_columns(
        state=pl.col("geography").replace(state_to_abbrv),
        log_mspe=pl.col("mspe").log(),
    )
)
mspe_mean_state = (
    mspe.filter(~((pl.col("season") == "2023/2024") & (pl.col("type") == "postcheck")))
    .group_by("geography")
    .agg(avg_log_mspe=pl.col("log_mspe").mean())
    .with_columns(
        state=pl.col("geography").replace(state_to_abbrv),
    )
)
mspe_mean_season = (
    mspe.filter(~((pl.col("season") == "2023/2024") & (pl.col("type") == "postcheck")))
    .group_by("season")
    .agg(avg_log_mspe=pl.col("log_mspe").mean())
)
mspe_mean_type = (
    mspe.filter(~((pl.col("season") == "2023/2024") & (pl.col("type") == "postcheck")))
    .group_by("type")
    .agg(avg_log_mspe=pl.col("log_mspe").mean())
)

# %% FIGURE 2A: Posterior check for PA in 2015/2016
plot_uptake(
    pred.filter(
        (pl.col("geography") == "Pennsylvania") & (pl.col("season") == "2015/2016")
    ).drop(["geography", "season"]),
    color=season_colors[6],
)
mspe.filter((pl.col("geography") == "Pennsylvania") & (pl.col("season") == "2015/2016"))

# %% FIGURE 2B: Posterior check for NV in 2017/2018
plot_uptake(
    pred.filter(
        (pl.col("geography") == "Nevada") & (pl.col("season") == "2017/2018")
    ).drop(["geography", "season"]),
    color=season_colors[8],
)
mspe.filter((pl.col("geography") == "Nevada") & (pl.col("season") == "2017/2018"))

# %% FIGURE 2C: Distribution of MSPE across states stratified by season
alt.Chart(mspe.filter(pl.col("season") != "2023/2024")).mark_point(
    filled=True, size=75, opacity=0.5
).encode(
    y=alt.Y(
        "log_mspe:Q",
        title="Log MSPE",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16),
        scale=alt.Scale(domain=[-11, -4]),
    ),
    x=alt.X(
        "season:N",
        title="Season",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
    ),
    color=alt.Color("season:N", scale=alt.Scale(range=season_colors)),
) + alt.Chart(mspe_mean_season.filter(pl.col("season") != "2023/2024")).mark_point(
    color="black", shape="square", size=100, opacity=1.0
).encode(
    y=alt.Y("avg_log_mspe:Q"),
    x=alt.X("season:N"),
)

# %% FIGURE 2D: Distribution of MSPE across seasons stratified by state
alt.Chart(mspe.filter(pl.col("season") != "2023/2024")).mark_point(
    filled=True, size=75, opacity=0.75
).encode(
    y=alt.Y(
        "log_mspe:Q",
        title="Log MSPE",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16),
        scale=alt.Scale(domain=[-11, -4]),
    ),
    x=alt.X(
        "state:N",
        title="State",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
    ),
    color=alt.Color("season:N", scale=alt.Scale(range=season_colors)),
) + alt.Chart(mspe_mean_state).mark_point(
    color="black", shape="square", size=100, opacity=1.0
).encode(
    y=alt.Y("avg_log_mspe:Q"),
    x=alt.X("state:N"),
)

# %% FIGURE 3A: Retrospective forecasting for Pennsylvania in 2023/2024
plot_uptake(
    pred.filter(
        (pl.col("geography") == "Pennsylvania") & (pl.col("season") == "2023/2024")
    )
    .drop(["geography", "season"])
    .with_columns(
        est=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est")),
        est_upper=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est_upper")),
        est_lower=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est_lower")),
    ),
    color=season_colors[14],
)

mspe.filter(
    (pl.col("geography") == "Pennsylvania")
    & (pl.col("season") == "2023/2024")
    & (pl.col("type") == "forecast")
)

# %% FIGURE 3B: Retrospective forecasting for Nevada in 2023/2024
plot_uptake(
    pred.filter((pl.col("geography") == "Nevada") & (pl.col("season") == "2023/2024"))
    .drop(["geography", "season"])
    .with_columns(
        est=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est")),
        est_upper=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est_upper")),
        est_lower=pl.when(pl.col("time_end").dt.month().is_in([7, 8]))
        .then(None)
        .otherwise(pl.col("est_lower")),
    ),
    color=season_colors[14],
)

mspe.filter(
    (pl.col("geography") == "Nevada")
    & (pl.col("season") == "2023/2024")
    & (pl.col("type") == "forecast")
)

# %% FIGURE 3C: Distribution of MSPE across states stratified by prediction type
alt.Chart(
    mspe.filter(~((pl.col("season") == "2023/2024") & (pl.col("type") == "postcheck")))
).mark_point(filled=True, size=75, opacity=0.5).encode(
    y=alt.Y(
        "log_mspe:Q",
        title="Log MSPE",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16),
        scale=alt.Scale(domain=[-11, -4]),
    ),
    x=alt.X(
        "type:N",
        title="Prediction Type",
        axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
        sort="descending",
    ),
    color=alt.Color("type:N", scale=alt.Scale(range=[season_colors[14], "gray"])),
) + alt.Chart(mspe_mean_type).mark_point(
    color="black", shape="square", size=100, opacity=1.0
).encode(
    y=alt.Y("avg_log_mspe:Q"),
    x=alt.X("type:N", sort="descending"),
)

# %% FIGURE 3D: Distribution of MSPE across prediction type stratified by state
(
    alt.Chart(mspe.filter(pl.col("season") != "2023/2024"))
    .mark_point(filled=True, size=75, opacity=0.75, color="gray")
    .encode(
        y=alt.Y(
            "log_mspe:Q",
            title="Log MSPE",
            axis=alt.Axis(titleFontSize=16, labelFontSize=16),
            scale=alt.Scale(domain=[-11, -4]),
        ),
        x=alt.X(
            "state:N",
            title="State",
            axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
        ),
    )
    + alt.Chart(mspe_mean_state)
    .mark_point(color="black", shape="square", size=100, opacity=1.0)
    .encode(
        y=alt.Y("avg_log_mspe:Q"),
        x=alt.X("state:N"),
    )
    + alt.Chart(
        mspe.filter((pl.col("season") == "2023/2024") & (pl.col("type") == "forecast"))
    )
    .mark_point(filled=True, size=75, opacity=1, color=season_colors[14])
    .encode(
        y=alt.Y(
            "log_mspe:Q",
            title="Log MSPE",
            axis=alt.Axis(titleFontSize=16, labelFontSize=16),
            scale=alt.Scale(domain=[-11, -4]),
        ),
        x=alt.X(
            "state:N",
            title="State",
            axis=alt.Axis(titleFontSize=16, labelFontSize=16, labelAngle=45),
        ),
    )
)
