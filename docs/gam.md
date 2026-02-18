# GAMs

## Overview

GAM (generalized additive model) predicts vaccination covarege using the number of days since vaccine roll-out date through spline function. A spline function is a smooth curve, composed by multiple polynomial functions, which are called basis functions, connected at certain points (knots). Denote vaccination coverage as $y_i$, elapsed days as $x_i$, spline function $f(.)$, the relation between $y_i$ and $x_i$ is:

```math

g^{-1}(E(y_i)) = f(x_i) + \beta_0

```

where $g^{-1}(.)$ is a link function and $\beta_0$ is population-level intercept.

Moreover, $f(x_i)$ is the sum of values evaluated at basis functions $B_k(.), k = 1,2,...$, weighted by coefficients $\beta_k$.

$$
f(x_i)=\sum_{k=1}^{K}{\beta_k}{B_k(x_{i})}
$$

Writing the formula in matrix form, this is:

```math

g^{-1}(E(y)) = X\beta + \beta_0

```

$y$ is a vector of observed vaccination coverage, $X$ is the design matrix of basis function with $N \times k$ dimension, where $N$ is the number of data points and $k$ is the number of basis functions used. The element in $X$ is the value of the basis function evaluated at the predictor elapsed, with rows are data point $x_i$ , and columns are basis function $B_k$.

$$
X_{i, k} = B_k(x_{i})
$$

**$\beta$** is a vector of coefficients that control how much influence each basis function has on the fit of spline function $f(x_i)$. It will be estimated in model fitting. We use identify link function for now, which is: $X\beta = E(y)$.

## Bayesian framework

We use Bayesian framework to fit the GAM model.

### Population-level effect

#### Model structure

$$
\begin{align}
& p(\beta,\lambda,\sigma^2 |y) ∝ p(y |\beta,\sigma^2)p(\beta|\lambda, \sigma^2)p(\lambda)p(\sigma^2) \\
& p(y|\beta, \sigma^2) \sim N(X\beta, \sigma^2I) \\
& p(\beta|\lambda, \sigma^2) \sim N(0, (\sigma^2/\lambda)S^{-}) \\
& p(\lambda) \sim Gamma(shape=1.0,rate=1.0) \\
& p(\sigma^2) \sim InverseGamma(shape=1.0,rate=1.0)
\end{align}
$$

#### Deriving prior of $\beta$

As we assume the link function is identity, that indicates the data $y$ follows normal distribution with covariance matrix $\sigma^2I$.

$$
log(y|\beta,\sigma^2) = -||y-X\beta||^2/(2\sigma^2) +c
$$

Instead of directly minimizing $||y - X\beta||$, we add a term to penalize the wiggliness of the spline. The target function to minimize becomes:

$$
\hat{\beta} = argmin\{||y-X\beta|| + \lambda \int f^{(k-1)}(x)^2 dx\}
$$

$f^{(k-1)}$ is the $(k-1)^{th}$ derivative of the spline function with order of $k$. $\int f^{(k-1)}(x)^2 dx$ measures the wiggleness of the spline function for the entire range of covariate $x$. $\lambda$ is a coefficient that controls how much penalization is imposed in this function, and will be estimated.

We can rewrite the function to minimize in pure matrix form:

$$
\hat{\beta} = argmin\{||y-X\beta|| + \lambda\beta^TS\beta\}
$$

where $S$ is called penalty matrix, and

$$
S_{ij} = \int{B''_i(x)B''_j(x)dx}
$$

Multiply $-\frac{1}{2\sigma^2}$, we have:

$$
\hat{\beta} = argmax\{-||y-X\beta||/(2\sigma^2) - \lambda\beta^TS\beta/(2\sigma^2)\}
$$

which is proportional to:

$$
log(y|\beta,\sigma^2) - \lambda\beta^TS\beta/(2\sigma^2)
$$

Exponentiating the function, we have:

```math
L(y|\beta)\cdot exp(-\lambda\beta^TS\beta/(2\sigma^2))
```

Using empirical Bayes approach, we can derive:

$$
\begin{align}
\hat{\beta} & ∝ L(y|\beta, \sigma^2) \cdot exp(-\lambda\beta^TS\beta/(2\sigma^2)) \\
\hat{\beta} & ∝ L(y|\beta,\sigma^2) \cdot exp(-\beta^T\beta/(2\sigma^2 /\lambda S))
\end{align}
$$

to have $\beta \sim N(0, (\sigma^2/\lambda)S^{-})$. This is the prior of $\beta$ and needs to be estimated from data.

## HGAMs

See [Pedersen et al.](https://peerj.com/articles/6876/) and [Wood](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-112723-034249).

Pedersen et al. write:

$$
\mathbb{E}[Y] = g^{-1}\left( \beta_0 + \sum_{j=1}^J f_j(x_j) \right)
$$

where $Y$ is the random variable representing the observed data, $g$ is the link function, $\beta_0$ is the grand mean, the index $j$ refers to covariates, and $f_j$ are the covariate smoothers.

Note that this model is "hierarchical" in the space of the _response_, that is, the sum inside $g^{-1}(\cdot)$. It is not hierarchical in the sense that most Bayesian models use the term, to imply that the model _parameters_ are added up in some way.

This is confusing because we want smoothers of a single covariate (time $t$), and we want different smoothers for the global trend (i.e., across seasons and states), for state-level trends, and for season-level trends. Technically, we would need to think of time as multiple covariates, replicated for each state and season. This is confusing. So instead write this as:

$$
\mathbb{E}[Y_{sg}(t)] = g^{-1}\left( \beta_0 + f_1(t) + f_{2s}(t) + f_{3g}(t) \right)
$$

where $f_1$ is the global smoother, $f_{2s}$ is the season-level smoother for season $s$, and $f_{3g}$ is the geography-level smoother for geography $g$.

Each smoother is a linear combination of basis functions:

$$
\begin{align*}
f_1(t) &= \sum_{k=1}^K \beta_{1k} b_{1k}(t) \\
f_{ij}(t) &= \sum_{k=1}^K \beta_{ijk} b_{ijk}(t) \\
\end{align*}
$$

where there are up to $K$ basis functions, the $\beta_{1k}$ are the coefficients for the global trend spline, and $\beta_{ijk}$ are the coefficients (e.g., $\beta_{234}$ would be the coefficient for the 4th basis function for the 3rd season), and the $b_{1k}$ and $b_{ijk}$ are the basis functions.

We also need some observation error model. For example, normally-distributed errors would be:

$$
L \equiv \prod_{tsg} \mathbb{P}\big[y_{tsg} \mid \mathbb{E}[Y_{tsg}] \big] = \mathcal{N}\left(y_{tsg}; \mathbb{E}[Y_{tsg}], \sigma_\varepsilon^2 \right)
$$

### Empirical Bayes

1. Pick basis functions, including the number and position of knots. An algorithm does this; in mgcv you specify the maximum number and it does the basis reduction, based on the data.
1. Compute the penalty matrices $\mathbf{S}_i$ from the basis functions $b_{ijk}$.
1. Choose smoothing parameters $\lambda_i$. An algorithm does this, based on the data.
1. Find the $\beta$, which includes $\beta_0$ and all the $\beta_{1k}$ and $\beta_{ijk}$, that minimizes $L - \sum_i \lambda_i \beta_i^T \mathbf{S}_i \beta_i$
