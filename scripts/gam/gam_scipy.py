import altair as alt
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import BSpline, make_lsq_spline

nis = pl.read_csv("data/nis_flu_national.csv")
nis = nis.filter(pl.col("season") == "2009/2010")

start_day = "-07-01"
nis = (
    nis.with_columns(
        roll_out=pl.col("season").str.split("/").list.get(0) + start_day
    ).with_columns(
        roll_out=pl.col("roll_out").str.strptime(pl.Date, "%Y-%m-%d"),
        time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"),
    )
).with_columns(elapsed=(pl.col("time_end") - pl.col("roll_out")).dt.total_days())


def get_cubic_spline(
    x: pl.Series, y: pl.Series, p: int = 3, num_internal_knots=6
) -> jnp.ndarray:
    """
    Get BSplineBasis object given covariate x, degree of basis spline function,
    and the number of internal knots (except boundary knots).
    The knots will be distributed as: [p*knots at x.min(), internal_knots, p*knots at x.max()],
    the internal knots are evenly spaced between the boundary knots.

    Args:
    x: pl.Series
        the value of covariate, here is "elapsed" from NIS data
    p: int
        the degree of basis spline function, here fix to 3, i.e, cubic spline function
    num_internal_knots: int
        the number of internal knots to be used

    Return:
    BSplineBasis object

    """
    # data
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # internal knots
    interior_knots = jnp.linspace(x.min(), x.max(), num_internal_knots + 2)[1:-1]

    # all knots
    t = jnp.concatenate(
        [jnp.repeat(x.min(), (p + 1)), interior_knots, jnp.repeat(x.max(), (p + 1))]
    )

    # build cubic spline between x and y
    cs = make_lsq_spline(x=x, y=y, t=t, k=p)

    return cs


def get_design_matrix(bs, t, x, p=3):
    Xobj = bs.design_matrix(x, t, p)
    X = jnp.array(Xobj.toarray())

    return X


def get_penalty_matrix(t, x, p=3):
    """
    Calculate the penalty matrix for cubic splines.

    The penalty matrix S has elements:
    S[i,j] = integral of (B_i''(x) * B_j''(x)) dx

    Args:
    knots : jnp.ndarray
        Knot vector for the cubic spline
    p : int
        Degree of the spline (3 for cubic spline)

    Return:
    S : ndarray
        Penalty matrix of shape (n_basis, n_basis)
    """
    # Number of basis functions: n = m - p - 1
    n_basis = len(t) - p - 1

    # Initialize penalty matrix
    S = jnp.zeros((n_basis, n_basis))

    # for each element, it is int(b_i"(x)*b_j"(x))dx
    for i in range(n_basis):
        for j in range(n_basis):
            # Create basis functions
            c_i = jnp.zeros(n_basis)
            c_i = c_i.at[i].set(1.0)
            # c_i[i] = 1.0
            c_j = jnp.zeros(n_basis)
            # c_j[j] = 1.0
            c_j = c_j.at[j].set(1.0)

            # Create B-spline objects
            spline_i = BSpline(t, c_i, p)
            spline_j = BSpline(t, c_j, p)

            # Second derivative
            spline_i_dd = spline_i.derivative(2)
            spline_j_dd = spline_j.derivative(2)

            # manual integration
            # S[i,j] = sum(spline_i_dd(x)*spline_j_dd(x))
            S = S.at[i, j].set(sum(spline_i_dd(x) * spline_j_dd(x)))

    return S


# numpyro model
# def gam(
#     S: ArrayLike,
#     X: ArrayLike,
#     estimate: ArrayLike = None,
#     sigma_z_rate=40,
#     groups=["season", "geography"],
#     num_group_factors=0,
#     num_group_levels=[0],
#     lam_shape=1.0,
#     lam_rate=1.0,
#     sigma_rate=40.0,
# ):
#     n, p = X.shape

#     # Priors
#     lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
#     sigma = numpyro.sample("sigma", dist.Exponential(sigma_rate))
#     # beta0 = numpyro.sample("beta0", dist.Normal(beta0_mean, beta0_sd))

#     if groups is not None:
#         sigma_z = numpyro.sample(
#             "sigma_z", dist.Exponential(sigma_z_rate), sample_shape=(num_group_factors,)
#         )

#         # Penalized precision matrix, add 1e-6 to make sure stability
#         precision = (lam * S) + 1e-6 * jnp.eye(p)

#         # Sample coefficients with constraint
#         mu_beta = numpyro.sample(
#             "mu_beta", dist.MultivariateNormal(0, sigma * jnp.linalg.inv(precision))
#         )

#         z = numpyro.sample(
#             "z",
#             dist.MultivariateNormal(0, jnp.eye(p)),
#             sample_shape=(sum(num_group_levels),),
#         )

#         # Compute mean
#         all_dev = z * jnp.repeat(sigma_z, jnp.array(num_group_levels))
#         beta_all = mu_beta + jnp.sum(all_dev)

#         mu = X @ beta_all

#         # Likelihood
#         numpyro.sample("obs", dist.Normal(mu, sigma), obs=estimate)
#     else:
#         raise NotImplementedError("At least one grouping factor: season is requried.")


# numpyro model
def gam(
    S: ArrayLike,
    X: ArrayLike,
    estimate: ArrayLike,
    lam_shape=1.0,
    lam_rate=1.0,
    sigma_rate=40.0,
    beta0_mean=0.0,
    beta0_sd=1.0,
):
    n, p = X.shape

    # Priors
    lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
    sigma = numpyro.sample("sigma", dist.Exponential(sigma_rate))
    beta0 = numpyro.sample("beta0", dist.Normal(beta0_mean, beta0_sd))

    # Penalized precision matrix, add 1e-6 to make sure stability
    precision = (lam * S) + 1e-6 * jnp.eye(p)

    # Sample coefficients with constraint
    beta = numpyro.sample(
        "beta",
        dist.MultivariateNormal(jnp.zeros(p), sigma * jnp.linalg.inv(precision)),
    )

    # Compute mean
    mu = X @ beta + beta0

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=estimate)


# compare posterior mean with data #
def posterior_mean(mcmc: MCMC) -> ArrayLike:
    samples = mcmc.get_samples()
    beta_sample = samples["beta"]
    beta_mean = jnp.mean(beta_sample, axis=0)
    beta0_mean = jnp.mean(samples["beta0"])

    mu = X @ beta_mean + beta0_mean

    return mu


if __name__ == "__main__":
    ## model fitting ##

    nis = nis.sort("elapsed")

    x = nis["elapsed"]
    y = nis["estimate"]

    bs = get_cubic_spline(x, y)

    X = get_design_matrix(bs, bs.t, x)
    print(X)

    S = get_penalty_matrix(bs.t, x)

    estimate = jnp.array(nis["estimate"])
    estimate = jnp.log(estimate + 1e-6)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(gam)
    mcmc_par = {"num_warmup": 1000, "num_samples": 1000, "num_chains": 4}
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_par["num_warmup"],
        num_samples=mcmc_par["num_samples"],
        num_chains=mcmc_par["num_chains"],
    )
    mcmc.run(rng_key_, S=S, X=X, estimate=estimate)

    ### model output ###

    # model summary #
    mcmc.print_summary()

    # compare posterior mean with data #
    mu = posterior_mean(mcmc)
    df = pl.DataFrame(
        {"data": nis["estimate"], "date": jnp.arange(len(nis["estimate"]))}
    )
    df = df.with_columns(mu=jnp.asanyarray(jnp.exp(mu)))

    charts = alt.Chart(df).mark_point().encode(x="date", y="data") + alt.Chart(
        df
    ).mark_line(color="red").encode(x="date", y="mu")

    charts.save("new_gam_fit.png")
