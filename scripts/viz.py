import altair as alt
import polars as pl
import streamlit as st


def app():
    st.title("Forecasts")

    # load forecast data
    data = load_data()

    # filter to season
    seasons = data["forecasts"]["season"].unique().to_list()
    season = st.selectbox("Season", options=seasons)

    # how many trajectories to show?
    n_samples = st.number_input("Number of forecasts", value=3, min_value=1)

    # filter the forecast data to a particular season, number of forecast
    # trajectories, and forecast date
    forecast_data = data["forecasts"].filter(
        pl.col("season") == pl.lit(season),
        pl.col("sample_id").cast(pl.Int32) <= n_samples,
    )
    forecast_dates = (
        forecast_data["forecast_start"].dt.truncate("1d").unique().to_list()
    )
    forecast_date = st.select_slider("Forecast date", options=forecast_dates)
    forecast_data = forecast_data.filter(
        pl.col("forecast_start").dt.truncate("1d") == forecast_date
    )

    forecast_chart = (
        alt.Chart(forecast_data)
        .mark_line()
        .encode(
            x=alt.X("time_end:T", title="Observation date"),
            y="estimate",
            color="model",
            detail="sample_id",
        )
    )

    # prepare the observed data and chart
    obs_data = data["nis_raw"].filter(pl.col("season") == pl.lit(season))

    obs_chart = (
        alt.Chart(obs_data)
        .mark_point()
        .encode(
            x=alt.X("time_end:T", title="Observation date"),
            y=alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        )
    )

    # combine and display chart
    chart = obs_chart + forecast_chart
    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def load_data():
    return {
        "forecasts": pl.read_parquet("data/forecasts.parquet"),
        "nis_raw": pl.read_parquet("data/nis_raw.parquet"),
    }


if __name__ == "__main__":
    app()
