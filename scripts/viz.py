import altair as alt
import polars as pl
import streamlit as st


def app():
    st.title("Forecasts")

    # load data
    data = load_data()

    # clean data
    forecast_data = (
        data["forecasts"]
        .with_columns(
            pl.col("forecast_start").dt.truncate("1d"),
            pl.col("sample_id").cast(pl.Int32),
        )
        .drop("forecast_end")
    )

    # get all forecast start dates and cross them with the observed data, as
    # if the observed data were a model
    forecast_starts = forecast_data.select(["season", "forecast_start"]).unique()

    obs_data = (
        data["nis_raw"]
        .with_columns(sample_id=0, model=pl.lit("observed"))
        .join(forecast_starts, on=["season"], how="inner")
        .select(forecast_data.columns)
    )

    data = pl.concat([forecast_data, obs_data], how="vertical")

    # set up plot encodings
    encodings = [
        alt.X("time_end:T", title="Observation date"),
        alt.Y("estimate:Q", title="Cumulative uptake estimate"),
        alt.Detail("sample_id", title="Trajectory"),
    ]

    # select which data dimension to put into which plot channel
    st.header("Plot options")
    st.subheader("Data channels")
    dimensions = ["season", "model", "forecast_start"]
    for channel, default_dim in [
        ("Color", "model"),
        ("Column", "season"),
        ("Row", "forecast_start"),
    ]:
        options = ["None"] + dimensions
        index = options.index(default_dim)
        dim = st.selectbox(f"{channel} by", options=options, index=index)

        if dim != "None":
            dimensions.remove(dim)
            encodings.append(getattr(alt, channel)(dim))

    # filter for specific values of the remaining dimensions
    st.header("Data filters")
    for dim in dimensions:
        filter_val = st.selectbox(dim, options=data[dim].unique().sort().to_list())
        data = data.filter(pl.col(dim) == pl.lit(filter_val))

    # how many trajectories to show?
    n_samples = st.number_input("Number of forecasts", value=3, min_value=1)
    data = data.filter(
        pl.col("sample_id").cast(pl.Int32) <= n_samples,
    )

    chart = alt.Chart(data).mark_line().encode(*encodings)

    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def load_data():
    return {
        "forecasts": pl.read_parquet("data/forecasts.parquet"),
        "nis_raw": pl.read_parquet("data/nis_raw.parquet"),
    }


if __name__ == "__main__":
    app()
