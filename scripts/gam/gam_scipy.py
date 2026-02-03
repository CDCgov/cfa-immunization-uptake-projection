import altair as alt
import jax.numpy as jnp
import numpy as np
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


def get_spline(x: pl.Series, y: pl.Series, p: int = 2, num_internal_knots=1) -> BSpline:
    """
    Get BSpline object given predictor xx-coordinates, response y, order of degree,
    and the number of internal knots.
    The knots will be distributed as: [p*knots at x.min(), internal_knots, p*knots at x.max()],
    the internal knots are evenly spaced between the boundary knots.

    Args:
    x: pl.Series
        the value of predictor x, must be sorted and unique
    y: pl.Series
        the value of response variable y
    p: int
        the degree of spline
    num_internal_knots: int
        the number of internal knots

    Return:
    BSpline object

    """
    # data
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    assert jnp.all(jnp.diff(x) >= 0), "x must not be descending."
    assert jnp.array_equal(x, jnp.unique(x)), "Each element in x must be unique."

    # internal knots
    interior_knots = jnp.linspace(x.min(), x.max(), num_internal_knots + 2)[1:-1]

    # all knots
    t = jnp.concatenate(
        [jnp.repeat(x.min(), (p + 1)), interior_knots, jnp.repeat(x.max(), (p + 1))]
    )

    # build spline
    bs = make_lsq_spline(x=x, y=y, t=t, k=p)

    return bs


def get_design_matrix(bs: BSpline, t: ArrayLike, x: pl.Series, p: int = 2) -> ArrayLike:
    """
    Get design matrix. Each element in design matrix is evaluation of basis function at x_i data point:
    X_ij = B_j(x_i).

    Args:
    bs: BSpline
        the BSpline object created based x and y
    t: ArrayLike
        the vector of knots
    x: pl.Series
        the values of predictor x

    Return:
    design matrix: ArrayLike

    """

    x = jnp.asarray(x)
    Xobj = bs.design_matrix(x, t, p)
    X = jnp.array(Xobj.toarray())

    return X


def get_penalty_matrix(t: ArrayLike, x: pl.Series, p: int) -> ArrayLike:
    """
    Get penalty matrix for spline functions, penalizing the smoothness.

    The smoothness is (p-1)th derivative for p degree of spline functions.

    The penalty matrix S has elements:
    S[i,j] = integral of (B_i^(p-1)(x) * B_j^(p-1)(x)) dx, where ^(p-1) means (p-1)th derivative

    Args:
    t: ArrayLike
        Knot vector for spline function
    x: pl.Series
        the data point to evaluate penalty
    p: int
        Degree of the spline

    Return:
    penalty matrix: ArrayLike
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
            spline_i_dd = spline_i.derivative(p - 1)
            spline_j_dd = spline_j.derivative(p - 1)

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
def posterior_mean(mcmc: MCMC, X) -> ArrayLike:
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

    p = 2
    num_internal_knots = 1
    bs = get_spline(x, y, p, num_internal_knots)

    X = get_design_matrix(bs, bs.t, x, p)

    S = get_penalty_matrix(bs.t, x, p)

    estimate = jnp.array(nis["estimate"])

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(gam)
    mcmc_par = {"num_warmup": 1000, "num_samples": 2000, "num_chains": 4}
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
    mu = posterior_mean(mcmc, X)
    df = pl.DataFrame({"data": nis["estimate"], "date": nis["time_end"]})
    df = df.with_columns(mu=np.asarray(mu))

    charts = alt.Chart(df).mark_point().encode(x="date", y="data") + alt.Chart(
        df
    ).mark_line(color="red").encode(x="date", y="mu")

    charts.save("new_gam_fit_09_10.png")
