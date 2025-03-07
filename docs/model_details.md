# Overview

This is a summary of the model structures used to capture and forecast vaccine uptake. There are currently two model structures: an autoregressive and a Hill function. Each model proposes a latent true uptake curve, which is subject to observation error. Each model also uses a hierarchical structure, in which parameters governing the latent true uptake curve for each group (defined by geography, age, and/or race) are drawn from a shared distribution representing a single season. The parameters governing that shared distribution are themselves drawn from a parent shared distribution across seasons.

# Notation

The following notation will be used across all models
- $t$ = number of days since rollout (i.e. $t_0$)
- $t_0$ = rollout, i.e. the last day before the vaccine was available
- $\hat{c}_t$ = observed cumulative uptake on day $t$
- $\hat{\sigma}_t$ = empirical estimate of the standard deviation of the observed cumulative uptake on day $t$
- $c_t$ = latent true cumulative uptake on day $t$
- $u_t$ = latent true incident uptake on day $t$
- $s$ = season (typically July 1 of a year until June 30 of the following year)
- $g$ = a combination of other grouping factors (e.g. geographic area, age group, race/ethnicity)

# Autoregressive Model

The autoregressive (AR) model is structured as follows:

```math
\begin{align*}
&\text{Observation Layer} \\
&\hat{c}_{t,s,g} \sim TruncNorm(c_{t,s,g}, \hat{\sigma}_{t,s,g}, 0, 1) \\
& \\
&\text{Functional Structure} \\
&c_{t,s,g} = \sum_{i=t_0}^{t} u_{i,s,g} \\
&u_{t,s,g} = \alpha_{s,g} + \beta_{s,g} \cdot u_{t-1,s,g} + \gamma_{s,g} \cdot t + \theta_{s,g} \cdot u_{t-1,s,g} \cdot t ~~~ \text{ for t > 0} \\
&u_{t_0,s,g} = 0 \\
& \\
&\text{Hierarchical Structure} \\
&\alpha_{s,g} \sim N(\alpha_s, \sigma_{\alpha, s}), ~ \beta_{s,g} \sim N(\beta_s, \sigma_{\beta, s}), ~ \gamma_{s,g} \sim N(\gamma_s, \sigma_{\gamma, s}), ~ \theta_{s,g} \sim N(\theta_s, \sigma_{\theta, s}) \\
&\alpha_s \sim N(\alpha, \sigma_{\alpha}), ~ \beta_s \sim N(\beta, \sigma_{\beta}), ~ \gamma_s \sim N(\gamma, \sigma_{\gamma}), ~ \theta_s \sim N(\theta, \sigma_{\theta}) \\
&\sigma_{\alpha, s} \sim Exp(\sigma_{\alpha}), ~ \sigma_{\beta, s} \sim Exp(\sigma_{\beta}), ~ \sigma_{\gamma, s} \sim Exp(\sigma_{\gamma}), ~ \sigma_{\theta,s} \sim Exp(\sigma_{\theta}) \\
& \\
&\text{Priors} \\
&\alpha,~\beta,~\gamma,~\theta \sim N(0, 0.1) \\
&\sigma_{\alpha},~\sigma_{\beta},~\sigma_{\gamma},~\sigma_{\theta} \sim Exp(0.1) \\
&t_0 \sim DiscreteUnif(\text{Earliest Day}, \text{ Latest Day})
\end{align*}
```

# Hill Model

The Hill model is structured as follows:

```math
\begin{align*}
\boldsymbol{\xi} &\sim \text{Pr}(\boldsymbol{\xi}) \\
\boldsymbol{\theta}_s &\sim \text{Pr}(\boldsymbol{\theta}_s \mid \boldsymbol{\xi}) \\
\boldsymbol{\theta}_{s, g} &\sim \text{Pr}(\boldsymbol{\theta}_{s,g} \mid \boldsymbol{\theta}_g) \\
\mathbf{c}_{s,g} &:= f_{\text{Hill}}(\boldsymbol{\theta}_{s, g}) \\
\hat{\mathbf{c}}_{s,g} &\sim \text{Pr}(\hat{\mathbf{c}}_{s,g} \mid \mathbf{c}_{s,g}, \hat{\boldsymbol{\sigma}}_{s,g})
\end{align*}
```

Where $\boldsymbol{\theta}_{s, g}$ are the parameters of the Hill models for each season and within each season for each combination of grouping factors.
More specicfically, we have the following model components.

## Observation Layer
```math
\begin{align*}
&\hat{c}_{t,s,g} \sim TruncNorm(c_{t,s,g}, \hat{\sigma}_{t,s,g}, 0, 1) \\
\end{align*}
```

## Functional Structure
The Hill function $f_{\text{Hill}}$ is,
```math
\begin{align*}
&\text{} \\
&c_{t,s,g} = \frac{A_{s,g} \cdot t^{n}}{H_{s,g}^{n} + t^{n}} \\
\end{align*}
```

## Hierarchical Structure
```math
\begin{align*}
&A_{s,g} \sim N(A_s, \sigma_{A, s}), ~ H_{s,g} \sim N(H_s, \sigma_{H, s}) \\
&A_s \sim N(A, \sigma_{A}), ~ H_s \sim N(H, \sigma_{H}) \\
&\sigma_{A, s} \sim Exp(\sigma_{A}), ~ \sigma_{H, s} \sim Exp(\sigma_{H}) \\
\end{align*}
```

## Priors
```math
\begin{align*}
&A \sim TruncNorm(0.4, 0.1, 0, 1), ~ H \sim TruncNorm(100,20, 0) \\
&\sigma_{A} \sim Exp(0.1), ~ \sigma_{H} \sim Exp(10)  \\
&n \sim Uniform(0.5, 4.5)
\end{align*}
```
