import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from scipy.interpolate import BSpline

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

# get design matrix from Scipy
x = np.linspace(30, 335, 100)
y = nis["elapsed"]


def get_design_matrix(x, y):
    # data
    x = np.asarray(x)  # shape (n,)
    y = np.asarray(y)

    # choose internal knots (example: quantiles)
    p = 3  # cubic
    m = 12  # number of basis function

    t = np.r_[
        np.repeat(x.min(), p + 1),
        np.quantile(x, np.linspace(0, 1, m - p + 1))[1:-1],
        np.repeat(x.max(), p + 1),
    ]

    X = BSpline.design_matrix(x, t, p).toarray()

    return X


# get penalty matrix
def get_penalty_matrix(X, p=3):
    ## given cubic spline, the penalty matrix is the squared second-order derivatives ##
    ## Sij = int{B_i''(x) B_j''(x) dx} = int{x_i}{x_j}
    None


# numpyro model
def gam(elapsed, estimate, lam_shape, lam_rate, sigma_shape, sigma_rate, S_inv):
    # sample the priors
    lam = numpyro.sample("lam", dist.Gamma(lam_shape, lam_rate))
    sigma = numpyro.sample("sigma", dist.InverseGamma(sigma_shape, sigma_rate))
    beta = numpyro.sample("beta", dist.Normal(0, S_inv * sigma / lam))

    # model formula
    mu = beta * elapsed

    # return the sample of observed estimate given mean
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=estimate)


mcmc = {"num_warmup": 1000, "num_samples": 1000, "num_chains": 2}
