# Overview

GAM (generalized additive model) models the relation between vaccine uptake ($y_i$) and the smooth version ($f(.)$) of elapsed variable (the number of days after vaccine roll-out) ($x_i$) and the random effect introduced by season ($u_j$) with link function $g^{-1}
(.)$.
```math
g^{-1}(E(y_i)) = f(x_i) + u_j + \beta_0

```

where $f(x_i)=\sum_{k=1}^{K}{\beta_k}{B_k(x_{i})}$.

$\beta_0$ is intercept for main effect.

Writing in matrix form, this is:

```math

g^{-1}(E(y)) = X\beta + Zu + \beta_0
```

$y$ is a vector of observed vaccine uptake, $X$ is the design matrix of basis function with $N \times k$ dimension, where $N$ is the number of observations and $k$ is the number of basis functions used. Each element in $X$ is the value of the basis function evaluated at the predictor elapsed ($B_k(x_{i})$).$\beta$ is a vector of coefficients that control each basis function. $X\beta$ is the main effect that is the same across all the level in a group (in our case is season). $Z$ is a random-effect design matrix to define the relation between levels in a group. $u$ is a vector representing season-specific intercept. Here, we assume the model follows lognormal distribution and the link function is $log()$.

## Bayesian framework

Because the main effect and the random effect are additive, we consider them separately for now.

### Main effect
The loglikelihood function is:

```math
Loglik(\beta, \lambda, u |y) = Loglik(y| \beta) - \lambda \beta^TS\beta
```
$S$ is called penalty matrix that is used to penalize the wiggliness of smooth function. In our case, we will use cubic spline function as the basis function, and the wiggliness of cubic spline function is measured as the integral of squared secondary derivatives of $B_k(x_{i})$, which is:

```math
S_{ij} = \int{B''_i(x)B''_j(x)dx}

```
In this way, $S$ penalizes the curvature of basis function. $\lambda$ is a smoothing parameter to control the balance between smoothness and fidelity of the data, which will be estimated along with $\beta$.

Exponentiating the loglikelhood function, we have:

```math
L(\beta,\lambda|y) = L(y|\beta)\cdot exp(-\lambda\beta^TS\beta)
```

Using empirical Bayes approach, we can derive:

$$
\begin{align}
L(\beta,\lambda|y) & ∝ L(y|\beta) \cdot exp(-\lambda\beta^TS\beta/(2\sigma^2)) \\
L(\beta,\lambda|y) & ∝ L(y|\beta) \cdot exp(-\beta^T\beta/(2\sigma^2 /\lambda S))
\end{align}
$$

to have a multivariate normal prior for $\beta$, where $\beta \sim N(0, S^{-1}\sigma^2/\lambda)$. The prior needs to be estimated from data.

Because it is assumed the model follows lognormal distribution, we have:

```math
log(y) \sim N(X\beta, \sigma^2I)

```

#### Model structure

$$
\begin{align}
& p(\beta,\lambda,\sigma^2 |y) ∝ p(y |\beta,\sigma^2)p(\beta|\lambda, \sigma^2)p(\lambda)p(\sigma^2) \\
& p(log(y)|\beta, \sigma^2) \sim N(X\beta, \sigma^2I) \\
& p(\beta|\lambda, \sigma^2) \sim N(0, S^{-1}\sigma^2/\lambda) \\
& p(\lambda) \sim Gamma(shape=1.0,rate=1.0) \\
& p(\sigma^2) \sim InverseGamma(shape=1.0,rate=1.0)
\end{align}
$$

#### Identifiability constraints

For each smooth term, it is possible to have identifiability issue between $f(x_i)$ and $\beta0$ when one of the basis function is also an intercept (first-order polynomials). Thus, it is needed to impose sum-to-zero constraint to ensure the identifiability of the smooth function, which is that the sum of the smooth function across the entire range of the covariate needs to be 0.

```math

\sum_i^N{f(x_i)} = 0

```
In matrix form, it is:

```math
1^TX\beta = 0
```

####
