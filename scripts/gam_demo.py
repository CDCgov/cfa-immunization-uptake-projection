import altair as alt
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization

# from scipy.interpolate import BSpline
from skfda.representation.basis import BSplineBasis

nis = pl.read_csv("data/nis_flu_national.csv")
start_day = "-07-01"
nis = (
    nis.with_columns(
        roll_out=pl.col("season").str.split("/").list.get(0) + start_day
    ).with_columns(
        roll_out=pl.col("roll_out").str.strptime(pl.Date, "%Y-%m-%d"),
        time_end=pl.col("time_end").str.strptime(pl.Date, "%Y-%m-%d"),
    )
).with_columns(elapsed=(pl.col("time_end") - pl.col("roll_out")).dt.total_days())

x = nis["elapsed"]


def get_Bspline_basis(
    x: pl.Series, p: int = 3, num_internal_knots: int = 6
) -> BSplineBasis:
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
    x = np.asarray(x)  # shape (n,)

    # Build B-spline basis with evenly spaced between boundaries.
    bss = BSplineBasis(
        order=p + 1,
        knots=list(np.quantile(x, np.linspace(0, 1, num_internal_knots + 2))[1:-1]),
    )

    return bss


def get_design_matrix(bss: BSplineBasis, x: pl.Series) -> np.ndarray:
    """
    Get the design matrix from BSplineBasis.

    Args:
    bss: BSplineBasis
        the BSplineBasis object used to evaluate at x
    x: pl.Series
        the value of covariate, here is "elapsed" from NIS data

    Return:
    np.ndarray
        the design matrix
    """
    # extract the matrix from 3-D np.array, and transpose to fit the N*m dimension of design matrix
    X = bss(x)[:, :, 0].T

    assert X.shape[0] == len(x), "Design matrix rows must match input length"

    # the number of basis function = num_internal_knots + 2*p(boundary knots) - p - 1
    assert X.shape[1] == len(bss.knots) + 2 * (bss.order - 1) - (bss.order - 1) - 1, (
        "Design matrix columns must match number of basis functions"
    )

    return X


def get_penalty_matrix(bss: BSplineBasis) -> np.ndarray:
    """
    Get the penalty matrix from BSplineBasis, given the fact that it is
    integrated secondary derivative of the design matrix.

    Args:
    bss: BSplineBasis
        the BSplineBasis object used to evaluate the penalty matrix

    Return:
    np.ndarray
        the penalty matrix
    """
    ## given cubic spline, the penalty matrix is the squared second-order derivatives ##
    ## Sij = int{B_i''(x) B_j''(x) dx} = int{x_i}{x_j}
    assert bss.order == 4, "Penalty matrix is only defined for cubic splines"

    derive_order = 2
    operator = LinearDifferentialOperator(derive_order)
    regularization = L2Regularization(operator)
    S = regularization.penalty_matrix(bss)

    expected_shape = len(bss.knots) + 2 * (bss.order - 1) - (bss.order - 1) - 1
    assert S.shape[0] == expected_shape, (
        "The row of penalty matrix must equal to number of basis function"
    )
    assert S.shape[1] == expected_shape, (
        "The column of penalty matrix must equal to number of basis function"
    )

    return S


# numpyro model
def gam(
    S: ArrayLike,
    X: ArrayLike,
    estimate: ArrayLike = None,
    sigma_z_rate=40,
    groups=["season", "geography"],
    num_group_factors=0,
    num_group_levels=[0],
    lam_shape=1.0,
    lam_rate=1.0,
    sigma_rate=40.0,
):
    n, p = X.shape

    # Priors
    lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
    sigma = numpyro.sample("sigma", dist.Exponential(sigma_rate))
    # beta0 = numpyro.sample("beta0", dist.Normal(beta0_mean, beta0_sd))

    if groups is not None:
        sigma_z = numpyro.sample(
            "sigma_z", dist.Exponential(sigma_z_rate), sample_shape=(num_group_factors,)
        )

        # Penalized precision matrix, add 1e-6 to make sure stability
        precision = (lam * S) + 1e-6 * jnp.eye(p)

        # Sample coefficients with constraint
        mu_beta = numpyro.sample(
            "mu_beta", dist.MultivariateNormal(0, sigma * jnp.linalg.inv(precision))
        )

        z = numpyro.sample(
            "z",
            dist.MultivariateNormal(0, jnp.eye(p)),
            sample_shape=(sum(num_group_levels),),
        )

        # Compute mean
        all_dev = z * jnp.repeat(sigma_z, jnp.array(num_group_levels))
        beta_all = mu_beta + jnp.sum(all_dev)

        mu = X @ beta_all

        # Likelihood
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=estimate)
    else:
        raise NotImplementedError("At least one grouping factor: season is requried.")


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

    bss = get_Bspline_basis(x=x)
    X = get_design_matrix(bss, x)
    S = get_penalty_matrix(bss)
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
        {"data": nis["estimate"], "date": np.arange(len(nis["estimate"]))}
    )
    df = df.with_columns(mu=np.asanyarray(jnp.exp(mu)))

    charts = alt.Chart(df).mark_point().encode(x="date", y="data") + alt.Chart(
        df
    ).mark_line(color="red").encode(x="date", y="mu")

    charts.save("gam_fit_k8.png")
