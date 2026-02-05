# GAMs

## Overview

GAM (generalized additive model) predicts vaccination covarege using the number of days since vaccine roll-out date through spline function. A spline function is a smooth curve, composed by multiple polynomial functions, which are called basis functions, connected at certain points (knots). Denote  vaccination coverage as $y_i$, elapsed days as $x_i$, spline function $f(.)$, the relation between $y_i$ and $x_i$ is:

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
