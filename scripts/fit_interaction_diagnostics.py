import pickle
from datetime import date

import altair as alt
import arviz as az
import polars as pl

alt.data_transformers.enable("vegafusion")

data = pl.read_parquet("../output/flu_state_all_inter/data.parquet")
print(
    len(data["season"].unique()),
    len(data["geography"].unique()),
    len(data["season_geo"].unique()),
)

# 13, 51, 663 # levels

with open("../output/flu_state_all_inter/fits/fit_2022-04-01.pkl", "rb") as f:
    fit = pickle.load(f)

fit = fit[("LPLModel", date(2022, 4, 1))]

az_obj = az.from_numpyro(fit.mcmc)

sliced = slice(64, 726)

# zA #
az.plot_posterior(az_obj, var_names=["zA"], coords={"zA_dim_0": sliced})
az.plot_trace(az_obj, var_names=["zA"], coords={"zA_dim_0": sliced})

# zM #
az.plot_posterior(az_obj, var_names=["zM"], coords={"zM_dim_0": sliced})
az.plot_trace(az_obj, var_names=["zM"], coords={"zM_dim_0": sliced})

pred = pl.read_parquet(
    "../output/flu_state_all_inter/pred/forecast_date=2022-04-01/part-0.parquet"
)
pred = pred.group_by(["time_end", "season", "geography", "season_geo"]).agg(
    pred_mean=pl.col("estimate").mean(),
    pred_lci=pl.col("estimate").quantile(0.975),
    pred_uci=pl.col("estimate").quantile(0.025),
)

all = data.join(pred, on=["time_end", "season", "geography", "season_geo"]).rename(
    {
        "estimate": "obs_mean",
        "lci": "obs_lci",
        "uci": "obs_uci",
    }
)

obs_chart = alt.Chart(all).mark_point().encode(x="t:Q", y="obs_mean:Q")

obs_ribbon_chart = (
    alt.Chart(all).mark_errorbar().encode(x="t:Q", y="obs_lci:Q", y2="obs_uci:Q")
)

pred_chart = (
    alt.Chart(all)
    .mark_line()
    .encode(
        x="t:Q",
        y="pred_mean:Q",
    )
)

pred_ribbon_chart = (
    alt.Chart(all)
    .mark_area(opacity=0.3)
    .encode(x="t:Q", y="pred_lci:Q", y2="pred_uci:Q")
)

alt.layer(obs_chart, obs_ribbon_chart, pred_chart, pred_ribbon_chart).facet(
    column="season:N", row="geography:N"
)
