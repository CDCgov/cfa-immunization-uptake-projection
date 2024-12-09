from typing import Any, Callable

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import polars as pl
import seaborn as sns
from numpyro.infer import MCMC, NUTS

# load data output from `fit_parametric_curves.R`
data = pl.read_csv("../output/nis2023.csv")
data.sample(5)


def model_factory(
    priors: dict[str, Any],
    curve: Callable[[np.array, dict[str, Any]], np.array],
):
    """Make a curve fitting model

    Given priors and a parametric curve, return a model, i.e., the kind of function that can be passed to
    the sampler. Output values from the curve are

    Args:
        priors (dict): mapping from variable name to numpyro.distributions object
        curve (function): (x,  sigma, pars) -> y, where:
          - x and y are arrays of the input and output curves
          - sigma is the standard deviation of each y value
          - pars is a dictionary of parameters, from string name to a numpyro distribution

    Returns:
        function to be called with MCMC(NUTS(...))
    """

    def model(x, sigma, y=None):
        # verify data inputs
        J = len(x)  # length of x input
        assert len(sigma) == J
        # the sampler sometimes calls this function without y values
        if y is not None:
            assert len(y) == J

        # turn prior names and distributions into numpyro.sample objects, which can be passed to curve functions
        pars = {name: numpyro.sample(name, distrib) for name, distrib in priors.items()}

        # get the point estimate position of the output curve
        theta = curve(x, pars)

        with numpyro.plate("J", J):
            numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)

    return model


# Curve functions -------------------------------------------------------------
def sigmoid(x, pars):
    """Sigmoidal curve"""
    return pars["ymax"] / (1.0 + jnp.exp(-pars["k"] * (x - pars["x0"])))


def hill(x, pars):
    """Hill equation

    Parameters are:
      xstart: before this time, output is zero
      xip_delay: time from xstart to inflection point
      mip: slope at inflection point
      ymax: final uptake
    """
    # derived parameters: inflection point, Hill coefficient
    xip = pars["xstart"] + pars["xip_delay"]
    n = 4 * xip * pars["mip"] / pars["ymax"]

    x_safe = jnp.where(x > pars["xstart"], x - pars["xstart"], 1.0)
    y = jnp.where(
        x > pars["xstart"],
        pars["ymax"] / (1 + (pars["xip_delay"] / x_safe) ** n),
        0.0,
    )

    return y


# Some example priors --------------------------------------------------------
sigmoid_priors = {
    "ymax": dist.Uniform(0, 100),
    "x0": dist.Uniform(-200, 200),
    "k": dist.Uniform(0, 10),
}

hill_priors = {
    "xstart": dist.Uniform(-50, 25),
    "xip_delay": dist.Uniform(0, 100),
    "mip": dist.Uniform(0, 100),
    "ymax": dist.Uniform(5, 50),
}

# Run an example -------------------------------------------------------------
# which curve and priors do you want to use?
my_curve = hill
my_priors = hill_priors

# build the model that's passed to the sampler
model = model_factory(
    priors=my_priors,
    curve=my_curve,
)

# set up the sampler
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=2000)

# set up random seed
rng_key = jax.random.PRNGKey(0)

# actually run the sampler
mcmc.run(
    rng_key,
    x=data["x"].to_numpy(),
    sigma=data["std"].to_numpy(),
    y=data["y"].to_numpy(),
)

# visualize the posterior samples
samples = pd.DataFrame(
    {key: value for key, value in mcmc.get_samples().items() if key != "obs"}
)
sns.pairplot(samples)

# visualize the posterior fits
fig, ax = plt.subplots()
ax.scatter(data["x"].to_numpy(), data["y"].to_numpy(), zorder=2)

for row in samples.sample(15).itertuples():
    row = row._asdict()
    x = np.linspace(-50, 200, num=100)
    y = my_curve(x, row)
    ax.plot(x, y, color="gray", zorder=1, alpha=0.5)

plt.show()
