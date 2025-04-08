import arviz as az
import pandas as pd
import polars as pl

import iup.models


def posterior_density_plot(idata: az.InferenceData) -> dict:
    return az.plot_posterior(idata)


def posterior_histogram_plot(idata: az.InferenceData) -> dict:
    return az.plot_dist(idata)


def parameter_trace_plot(idata: az.InferenceData) -> dict:
    return az.plot_trace(idata)


def parameter_pairwise_plot(idata: az.InferenceData) -> dict:
    return az.plot_pair(idata)


def print_posterior_dist(
    fit_model: iup.models.UptakeModel, idata: az.InferenceData
) -> pl.DataFrame:
    posterior = idata.to_dataframe(groups="posterior", include_coords=False)
    posterior = pl.from_pandas(posterior)
    # Rename columns using the actual levels of grouping factors, not numeric codes
    if fit_model.value_to_index is not None:
        group_factors = list(fit_model.value_to_index.keys())
        group_levels = [
            k
            for inner_dict in fit_model.value_to_index.values()
            for k in inner_dict.keys()
        ]
        group_factors_dict = {
            "[" + str(i) + "]": "_" + v.replace(" ", "_")
            for i, v in enumerate(group_factors)
        }
        group_levels_dict = {
            "[" + str(i) + "]": "_" + v.replace(" ", "_")
            for i, v in enumerate(group_levels)
        }
        for k, v in group_factors_dict.items():
            posterior = posterior.rename(
                {
                    col: col.replace(k, v) if "sigs" in col else col
                    for col in posterior.columns
                }
            )
        for k, v in group_levels_dict.items():
            posterior = posterior.rename(
                {
                    col: col.replace(k, v) if "devs" in col else col
                    for col in posterior.columns
                }
            )
    return posterior


def print_model_summary(idata: az.InferenceData) -> pd.DataFrame:
    return az.summary(idata)
