from typing import List

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS
from scipy.interpolate import BSpline, make_lsq_spline

nis = pl.read_parquet("data/raw.parquet")
nis.head()

start_day = "-07-01"
nis = (
    nis.with_columns(
        roll_out=pl.col("season").str.split("/").list.get(0) + start_day
    ).with_columns(
        roll_out=pl.col("roll_out").str.strptime(pl.Date, "%Y-%m-%d"),
        time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"),
    )
).with_columns(elapsed=(pl.col("time_end") - pl.col("roll_out")).dt.total_days())


def get_spline(x: ArrayLike, y: ArrayLike, p: int = 2, num_internal_knots=1) -> BSpline:
    """
    Get BSpline object given predictor xx-coordinates, response y, order of degree,
    and the number of internal knots.
    The knots will be distributed as: [p*knots at x.min(), internal_knots, p*knots at x.max()],
    the internal knots are evenly spaced between the boundary knots.

    Args:
    x: ArrayLike
        the value of predictor x, must be sorted and unique
    y: ArrayLike
        the value of response variable yS
    p: int
        the degree of spline
    num_internal_knots: int
        the number of internal knots

    Return:
    BSpline object

    """

    # internal knots
    interior_knots = jnp.linspace(x.min(), x.max(), num_internal_knots + 2)[1:-1]

    # all knots
    t = jnp.concatenate(
        [jnp.repeat(x.min(), (p + 1)), interior_knots, jnp.repeat(x.max(), (p + 1))]
    )

    # build spline
    bs = make_lsq_spline(x=x, y=y, t=t, k=p)

    return bs


def get_design_matrix(bs: BSpline, t: ArrayLike, x: ArrayLike, p: int = 2) -> ArrayLike:
    """
    Get design matrix. Each element in design matrix is evaluation of basis function at x_i data point:
    X_ij = B_j(x_i).

    Args:
    bs: BSpline
        the BSpline object created based x and y
    t: ArrayLike
        the vector of knots
    x: ArrayLike
        the values of predictor x

    Return:
    design matrix: ArrayLike

    """
    Xobj = bs.design_matrix(x, t, p)
    X = jnp.array(Xobj.toarray())

    return X


def get_penalty_matrix(t: ArrayLike, x: ArrayLike, p: int = 2) -> ArrayLike:
    """
    Get penalty matrix for spline functions, penalizing the smoothness.

    The smoothness is (p-1)th derivative for p degree of spline functions.

    The penalty matrix S has elements:
    S[i,j] = integral of (B_i^(p-1)(x) * B_j^(p-1)(x)) dx, where ^(p-1) means (p-1)th derivative

    Args:
    t: ArrayLike
        Knot vector for spline function
    x: ArrayLike
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

            c_j = jnp.zeros(n_basis)
            c_j = c_j.at[j].set(1.0)

            # Create B-spline objects
            spline_i = BSpline(t, c_i, p)
            spline_j = BSpline(t, c_j, p)

            # Second derivative
            spline_i_dd = spline_i.derivative(p - 1)
            spline_j_dd = spline_j.derivative(p - 1)

            # manual integration
            S = S.at[i, j].set(sum(spline_i_dd(x) * spline_j_dd(x)))

    return S


# numpyro model
def gam(
    data: pl.DataFrame,
    p: int = 2,
    num_internal_knots: int = 1,
    sigma_z_rate=40,
    groups=["season", "geography"],
    num_group_factors=0,
    num_group_levels: ArrayLike = jnp.array([0]),
    lam_shape=1.0,
    lam_rate=1.0,
    sigma_rate=40.0,
):
    # Priors
    lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
    sigma = numpyro.sample("sigma", dist.Exponential(sigma_rate))

    if groups is None:
        # Get the design matrix
        y = jnp.asarray(data["estimate"])
        x = jnp.asarray(data["elapsed"])

        bs = get_spline(x, y, p, num_internal_knots)

        X = get_design_matrix(bs, bs.t, x, p)
        S = get_penalty_matrix(bs.t, x, p)

        # Penalized precision matrix, add 1e-6 to make sure stability
        n, k = X.shape
        precision = (lam * S) + 1e-6 * jnp.eye(k)

        # Sample coefficients with constraint, this is mean beta vector across all groups
        beta = numpyro.sample(
            "beta",
            dist.MultivariateNormal(jnp.zeros(k), sigma * jnp.linalg.inv(precision)),
        )

        # Compute mean
        mu = X @ beta
        # Likelihood
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    else:
        data = index_combo(data, groups)

        sigma_z = numpyro.sample(
            "sigma_z", dist.Exponential(sigma_z_rate), sample_shape=(num_group_factors,)
        )

        mu_all = jnp.array([])
        y_all = jnp.asarray(data["estimate"])

        for idx in data["group_combo_idx"].unique():
            single_subset = data.filter(pl.col("group_combo_idx") == idx)

            x = jnp.asarray(single_subset["elapsed"])

            y = jnp.asarray(single_subset["estimate"])
            bs = get_spline(x, y)
            X = get_design_matrix(bs, bs.t, x)

            S = get_penalty_matrix(bs.t, x)

            n, k = X.shape

            z = numpyro.sample(
                f"z_{idx}",
                dist.MultivariateNormal(0, jnp.eye(k)),
                sample_shape=(sum(num_group_levels),),
            )

            dev = (
                z * jnp.repeat(sigma_z, num_group_levels)[:, None]
            )  # broadcast to allowing multiplying with z

            # Penalized precision matrix, add 1e-6 to make sure stability
            precision = (lam * S) + 1e-6 * jnp.eye(k)

            # Sample coefficients with constraint, this is mean beta vector across all groups
            beta = numpyro.sample(
                f"beta_{idx}",
                dist.MultivariateNormal(
                    jnp.zeros(k), sigma * jnp.linalg.inv(precision)
                ),
            )

            beta_all = beta + jnp.sum(dev, axis=0)
            mu = X @ beta_all

            mu_all = jnp.concat([mu_all, mu])

        # Likelihood
        numpyro.sample("obs", dist.Normal(mu_all, sigma), obs=y_all)


def index_combo(data: pl.DataFrame, groups: List):
    data = data.with_columns(
        group_combo=pl.concat_str([pl.col(g) for g in groups], separator="_")
    )
    unique_values = data["group_combo"].unique().sort().to_list()
    indices = list(range(len(unique_values)))

    replace_map = {value: index for value, index in zip(unique_values, indices)}

    data = data.with_columns(
        group_combo_idx=pl.col("group_combo").replace_strict(replace_map)
    ).drop("group_combo")

    return data


if __name__ == "__main__":
    ## model fitting ##

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # use a vignette to test model #
    data = pl.read_parquet("data/raw.parquet")
    data = data.filter(
        pl.col("season").is_in(["2020/2021", "2019/2020"]),
        pl.col("geography").is_in(["Alabama", "Alaska"]),
    )

    kernel = NUTS(gam)
    mcmc_par = {"num_warmup": 1000, "num_samples": 2000, "num_chains": 4}
    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_par["num_warmup"],
        num_samples=mcmc_par["num_samples"],
        num_chains=mcmc_par["num_chains"],
    )

    groups = ["season", "geography"]
    num_group_levels = jnp.array([2, 2])

    mcmc.run(
        rng_key_,
        p=2,
        groups=groups,
        num_group_factors=len(groups),
        num_group_levels=num_group_levels,
        data=data,
    )

    # model summary #
    mcmc.print_summary()
