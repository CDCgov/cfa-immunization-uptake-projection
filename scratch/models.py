import jax
import numpyro.util
import numpyro.distributions as dist
import jax.numpy as jnp

def sim_cartoon_uptake_data(seed, grid_times, midpoint, growth_rate, max_uptake = 0.3, obs_sigma=0.01):
    """
    Latent true uptake at time t U(t) = max_uptake / (1 + exp(-growth_rate * (t - midpoint)))

    True incident uptake between grid_times[i] and grid_times[i + 1] is U(grid_times[i + 1] - grid_times[i])

    Observations are drawn from a truncated normal with mean given by truth and SD = obs_sigma
    """
    true_cumulative = max_uptake / (1.0 + jnp.exp(-growth_rate * (jnp.array(grid_times) - midpoint)))
    true_incident = jnp.diff(true_cumulative)
    with numpyro.handlers.seed(rng_seed=seed):
        obs_incident = numpyro.sample("incident_uptake", dist.TruncatedNormal(loc=true_incident, scale=obs_sigma, low=0.0, high=1.0))
    return obs_incident


def logistic_incident(grid_times):
    # Should probably not hard-code any of these hyperparameters

    # Somewhat reasonable, somewhat informative uptake prior
    max_uptake = numpyro.sample(
        "max_uptake",
        dist.Beta(1.5, 3.5)
    )

    # This is probably a terrible prior, both in functional form and
    # in hyperparameter chosen, growth rates aren't intuitive to me
    growth_rate = numpyro.sample(
        "growth-rate",
        dist.Exponential(1)
    )

    # Definitely very bad but probably easy-ish to fix
    midpoint = numpyro.sample(
        "midpoint",
        dist.Uniform(-365, 365)
    )
    
    latent_cumulative = max_uptake / (1.0 + jnp.exp(-growth_rate * (jnp.array(grid_times) - midpoint)))

    latent_incident = numpyro.deterministic(
        "latent_incident",
        jnp.diff(latent_cumulative)
    )

    return latent_incident

def iid_trunc_normal_loglik(latent_incident, observed_incident, obs_idx = None):
    if obs_idx is None:
        obs_idx = jnp.arange(len(latent_incident)).astype(int)
    # Shouldn't hard code this hyperparam either
    sigma = numpyro.sample(
        "obs_sigma",
        dist.Exponential(10)
    )

    numpyro.sample(
        "obs_incident_uptake",
        dist.Normal(loc=latent_incident[obs_idx], scale=sigma),
        obs=observed_incident[obs_idx],
    )

    if len(obs_idx) < len(latent_incident):
        mask = jnp.zeros(len(latent_incident), dtype=bool)
        mask[obs_idx] = True
        unobs_idx = jnp.where(~mask)[0]
        numpyro.sample(
            "unobs_incident_uptake",
            dist.Normal(loc=latent_incident[unobs_idx], scale=sigma),
        )

def joint_model(prior, likelihood, grid_times, observed_incident, obs_idx = None):
    latent_incident = prior(grid_times)
    likelihood(latent_incident, observed_incident, obs_idx=obs_idx)