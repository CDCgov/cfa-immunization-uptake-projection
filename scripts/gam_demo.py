import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro import MCMC, NUTS
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regulations import L2Regularization

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

print((nis["elapsed"].min(), nis["elapsed"].max()))

x = nis["elapsed"]


def get_Bspline_basis(
    x: pl.Series, p: int = 3, num_internal_knots: int = 8
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
    S=None,
    X=None,
    estimate=None,
    lam_shape=1.0,
    lam_rate=1.0,
    sigma_shape=1.0,
    sigma_rate=1.0,
    beta_mu=0,
):
    # sample the priors
    lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
    sigma = numpyro.sample("sigma", dist.InverseGamma(sigma_shape, sigma_rate))

    beta_mu_array = jnp.array([beta_mu] * S.shape[0])
    S_inv = jnp.linalg.inv(S)

    beta = numpyro.sample(
        "beta", dist.MultivariateNormal(beta_mu_array, S_inv * sigma / lam)
    )

    # Identifiability constraints: 1^TX\beta = 0
    Identity = jnp.array([1.0] * X.shape[0])

    if Identity @ X @ beta.T == 0.0:
        # model formula
        mu = X @ beta.T

        # return the sample of observed estimate given mean
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=estimate)


bss = get_Bspline_basis(x=x)
X = get_design_matrix(bss, x)
S = get_penalty_matrix(bss)
estimate = nis["estimate"].to_numpy()

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

kernel = NUTS(gam)
mcmc_par = {"num_warmup": 1000, "num_samples": 1000, "num_chains": 1}
mcmc = MCMC(
    kernel,
    num_warmup=mcmc_par["num_warmup"],
    num_samples=mcmc_par["num_samples"],
    num_chains=mcmc_par["num_chains"],
)
mcmc.run(rng_key_, S=S, X=X, estimate=estimate)
