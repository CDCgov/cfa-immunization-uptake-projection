# Overview

GAM (generalized additive model) models the relation between vaccine uptake ($y_i$) and the smooth version ($f(.)$) of elapsed variable (the number of days after vaccine roll-out) ($x_i$) and the random effect introduced by season ($u_j$).

```math
y_i = f(x_i) + u_j + \beta_0

```

where $f(x_i)=\sum_{k=1}^{K}{\beta_k}{B_k(x_{i,k})}$.

$\beta_0$ is intercept for main effect.

Writing in matrix form, this is:

```math

y = X\beta + Z\u + \beta_0
```

$y$ is a vector of observed vaccine uptake, $X$ is the design matrix of basis function with $N \times k$ dimension, where $N$ is the number of observations and $k$ is the number of basis functions used. Each element in $X$ is the value of the basis function evaluated at the predictor elapsed ($B_k(x_{i,k})$).$\beta$ is a vector of coefficients that control each basis function. $X\beta$ is the main effect that is the same across all the level in a group (in our case is season).

The loglikelihood function is:

```math
Loglik(\beta, \lambda |y) = Loglik(y| \beta) - \lambda \beta^TS\beta
```
$S$ is called penalty matrix that is used to penalize the wiggliness of smooth function. In our case, we will use cubic spline function as the basis function, and the wiggliness of cubic spline function is measured as the integral of squared secondary derivatives of $B_k(x_{i,k})$, which is:

```math
S_{ij} = \int{B''_i(x)B''_j(x)dx}

```
In this way, $S$ penalizes the curvature of basis function. $\lambda$ is a smoothing parameter to control the balance between smoothness and fidelity of the data, which will be estimated along with $\beta$.

## Bayesian framework

### Main effect

The parameters to estimate are: $\beta, \lambda$. Exponentiating the loglikelhood function, we have:

```math
L(\beta,\lambda) = L(\beta)\cdot exp(-\lambda\beta^TS\beta)
```

We can empirically derive:

$$
\begin{align}
L(\beta,\lambda) & ∝ L(\beta) \cdot exp(-\lambda\beta^TS\beta/(2\sigma^2)) \\
L(\beta,\lambda) & ∝ L(\beta) \cdot exp(-\beta^T\beta/(2\sigma^2 /\lambda S))
\end{align}
$$

to have a multivariate normal prior for $\beta$, where $\beta \sim N(0, S^{-1}\sigma^2/\lambda)$

For likelihood function $L(y|\beta,\lambda)$, we have:

```math
loglik(y|\beta,\lambda) = -||y-X\beta||^2/(2\sigma^2) - \lambda\beta^TS\beta/(2\sigma^2)+c,
```
The RHS is maximized by $\hat{\beta}$, and can be replaced by its Taylor expansion about $\hat{\beta}$, thus:
```math
loglik(y|\beta,\lambda)=loglik(y, \hat{\beta}) -(\beta-\hat{\beta})^T(X^TX+\lambda S)(\beta-\hat{\beta}) + c

```
We can see $\beta |y \sim N(\hat{\beta},(X^TX+\lambda S)^{-1}) \sigma^2$

#### Priors

$$
\begin{align}
&\beta \sim N(0, S^{-1}\sigma^2/\lambda) \\
&\lambda \sim Gamma(shape=1.0,rate=10.0) \\
&\sigma \sim Exponential(rate=40.0) \\
\end{align}
$$
