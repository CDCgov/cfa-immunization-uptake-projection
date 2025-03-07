# Overview

This is a summary of the model structures used to capture and forecast vaccine uptake. There are currently three model structures: autoregressive, Hill, and hypertabastic. Each model proposes a latent true uptake curve, which is subject to observation error. Each model also uses a hierarchical structure, in which parameters governing the latent true uptake curve for each group (defined by geography, age, and/or race) are drawn from a shared distribution representing a single season. The parameters governing that shared distribution are themselves drawn from a parent shared distribution across seasons.

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
&\sigma_{\alpha},~\sigma_{\beta},~\sigma_{\gamma},~\sigma_{\theta} \sim Exp(0,0.1) \\
&t_0 \sim DiscreteUnif(\text{Earliest Day}, \text{ Latest Day})
\end{align*}
```
