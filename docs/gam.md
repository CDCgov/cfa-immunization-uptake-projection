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

g^{-1}(E(y)) = X\beta

```

$y$ is a vector of observed vaccination coverage, $X$ is the design matrix of basis function with $N \times k$ dimension, where $N$ is the number of data points and $k$ is the number of basis functions used. $k$ is defined by the order degree of spline function $p$ and the number of internal knots $m$ by:

$$
k = m + p - 1
$$

The element in $X$ is the value of the basis function evaluated at the predictor elapsed, with rows are data point $x_i$ , and columns are basis function $B_k$.

$$
X_{i, k} = B_k(x_{i})
$$

**$\beta$** is a vector of coefficients that control how much influence each basis function has on the fit of spline function $f(x_i)$. It will be estimated in model fitting. We assume $y$ is normally distributed and use identify link function for now, which is: $X\beta = E(y)$.

## Bayesian framework

We use Bayesian framework to fit the GAM model.

### Population-level effect

#### Model structure

```math
\begin{align*}
p(\beta,\lambda,\sigma |y) & ∝ p(y |\beta,\sigma)p(\beta|\lambda, \sigma)p(\lambda)p(\sigma) \\
p(y|\beta, \sigma) & \sim MultiNormal(X\beta, \sigma I) \\
p(\beta|\lambda, \sigma) & \sim MultiNormal(0, (\sigma/\lambda)S^{-}) \\
p(\lambda) & \sim Gamma(shape=1.0,rate=1.0) \\
p(\sigma) & \sim Exp(40)
\end{align*}
```


#### Deriving prior of $\beta$

As we assume the link function is identity, that indicates the data $y$ follows normal distribution with covariance matrix $\sigma I$.
$$
log(y|\beta,\sigma) = -||y-X\beta||^2/(2\sigma) +c
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

Multiply $-\frac{1}{2\sigma}$, we have:

$$
\hat{\beta} = argmax\{-||y-X\beta||/(2\sigma) - \lambda\beta^TS\beta/(2\sigma)\}
$$
which is proportional to:
$$
log(y|\beta,\sigma) - \lambda\beta^TS\beta/(2\sigma)
$$

Exponentiating the function, we have:

```math
L(y|\beta)\cdot exp(-\lambda\beta^TS\beta/(2\sigma))
```

Using empirical Bayes approach, we can derive:

$$
\begin{align}
\hat{\beta} & ∝ L(y|\beta, \sigma) \cdot exp(-\lambda\beta^TS\beta/(2\sigma)) \\
\hat{\beta} & ∝ L(y|\beta,\sigma) \cdot exp(-\beta^T\beta/(2\sigma /\lambda S))
\end{align}
$$

to have $\beta \sim N(0, (\sigma/\lambda)S^{-})$. This is the prior of $\beta$ and needs to be estimated from data.

### Group effect

The group factors are season ($s$) and geography ($g$). Their effects are introduced in $\beta$, to allow varying shape of the spline function adjusted by each $\beta_k$ to control the corresponding basis function $B_k$.

Given the population mean of $\beta$, denoted as $\bar \beta$, the $\beta$ specific to a certain season ($s=i$) and certain geography ($g=j$) is $\bar \beta$ plus vector $\delta_{s=i}$ and $\delta_{g=j}$. $\delta_{s=i}$ defines the deviation of the certain season from $\bar \beta$ and the certain geography from $\bar \beta$.

#### Model structure

```math
\begin{align*}
\beta_{total} &= \bar \beta + \delta_{s=i} + \delta_{g=j} \\
\mu &= X\beta_{total} \\
Y &\sim N(\mu, \sigma) \\

\bar \beta &\sim MultiNormal(0, (\sigma/\lambda)S^{-}) \\
\delta_s &\sim MultiNormal(0, \sigma_s I) \\
\delta_g &\sim MultiNormal(0, \sigma_g I) \\
\lambda &\sim Gamma(shape=1.0,rate=1.0) \\
\sigma_s &\sim Exp(40) \\
\sigma_g &\sim Exp(40) \\
\sigma &\sim Exp(40)
\end{align*}
```
