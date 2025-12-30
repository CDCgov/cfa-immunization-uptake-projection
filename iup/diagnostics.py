import arviz as az
import polars as pl

import iup.models


def posterior_density_plot(model: iup.models.UptakeModel) -> dict:
    idata = az.from_numpyro(model.mcmc)
    return az.plot_posterior(idata)


def parameter_trace_plot(model: iup.models.UptakeModel) -> dict:
    idata = az.from_numpyro(model.mcmc)
    return az.plot_trace(idata)


def parameter_pairwise_plot(model: iup.models.UptakeModel) -> dict:
    idata = az.from_numpyro(model.mcmc)
    if isinstance(model, iup.models.LPLModel):
        # remove A_devs and M_devs for LPL model
        return az.plot_pair(idata, var_names=["~A_devs", "~M_devs"])
    else:
        return az.plot_pair(idata)


def print_posterior_dist(model: iup.models.UptakeModel) -> pl.DataFrame:
    idata = az.from_numpyro(model.mcmc)
    posterior = idata.to_dataframe(groups="posterior", include_coords=False)
    posterior = pl.from_pandas(posterior)

    # Rename columns using the actual levels of grouping factors, not numeric codes
    if isinstance(model, iup.models.LPLModel):
        if model.value_to_index is not None:
            group_factors = list(model.value_to_index.keys())
            group_levels = [
                k
                for inner_dict in model.value_to_index.values()
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


def print_model_summary(model: iup.models.UptakeModel) -> pl.DataFrame:
    idata = az.from_numpyro(model.mcmc)
    summary_pd = az.summary(idata)
    summary = pl.DataFrame(summary_pd)
    summary = summary.with_columns(params=pl.Series(summary_pd.index)).select(
        ["params"] + [col for col in summary.columns if col != "params"]
    )

    return summary
