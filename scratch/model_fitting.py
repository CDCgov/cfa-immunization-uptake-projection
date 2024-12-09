import numpyro
import jax.numpy as jnp
from scratch.models import sim_cartoon_uptake_data, joint_model, logistic_incident, iid_trunc_normal_loglik
from numpyro.infer import MCMC, NUTS
from jax.random import key as jr_key

numpyro.set_host_device_count(4)

grid_times = jnp.array(list(range(0, 7*16, 7)))

inc_upt = sim_cartoon_uptake_data(42, grid_times=grid_times, midpoint=28, growth_rate=0.075)

mcmc = MCMC(
    NUTS(joint_model),
    num_samples=1000,
    num_warmup=1000,
    num_chains=4,
)

mcmc.run(jr_key(0), logistic_incident, iid_trunc_normal_loglik, grid_times, inc_upt[:6], jnp.arange(6))

mcmc.print_summary()

mcmc.get_samples()
