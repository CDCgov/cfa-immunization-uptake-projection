import arviz as az


def posterior_density_plot(idata: az.InferenceData):
    return az.plot_posterior(idata)


def posterior_histogram_plot(idata: az.InferenceData):
    return az.plot_dist(idata)


def parameter_trace_plot(idata: az.InferenceData):
    return az.plot_trace(idata)


def parameter_pairwise_plot(idata: az.InferenceData):
    return az.plot_pair(idata)


def print_posterior_dist(idata: az.InferenceData):
    return idata.to_dataframe(groups="posterior", include_coords=False)


def print_model_summary(idata: az.InferenceData):
    return az.summary(idata)
