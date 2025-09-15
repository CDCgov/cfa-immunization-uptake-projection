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
                color=alt.Color(
                    "season:N", scale=alt.Scale(scheme="category20")
                ),  # "season:N",
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

# %% Figure 1a: National flu vaccine uptake by season
plot_uptake(flu_natl, season_start_month=7)

# %% Figure 1b: State flu vaccine uptake by season
plot_uptake(
    flu_state.filter(pl.col("geography") == "Nevada").drop("geography"),
    season_start_month=7,
    upper_bound=0.7,
)
plot_uptake(
    flu_state.filter(pl.col("geography") == "Rhode Island").drop("geography"),
    season_start_month=7,
    upper_bound=0.7,
)


# %% Figure 1c: Avg. vs. std. dev. of final uptake by state
may31 = (
    flu_state.filter(pl.col("time_end").dt.month() == 5)
    .group_by("geography")
    .agg(avg=pl.col("obs").mean(), std=pl.col("obs").std())
    .with_columns(
        state=pl.col("geography").replace(state_to_abbrv),
    )
)
alt.Chart(may31).mark_text(align="center", baseline="middle", fontSize=10).encode(
    x=alt.X("avg:Q", title="Average", scale=alt.Scale(domain=[0.15, 0.55])),
    y=alt.Y("std:Q", title="Standard Deviation", scale=alt.Scale(domain=[0.02, 0.09])),
    text="state:N",
)

# %%
