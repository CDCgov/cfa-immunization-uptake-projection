# Overview

This is a summary of the models used to capture and forecast vaccine uptake. There are currently two models: an autoregressive and a mixed Hill/linear function. Each model proposes a latent true uptake curve, which is subject to observation error. Each model uses a hierarchy to account for grouping factors (e.g. season, geography, age, and/or race). In particular, the parameters governing the latent true uptake curve for each group (a combination of one value of each grouping factor) are given by baseline values plus deviations drawn from distributions representing each grouping factor.

# Autoregressive Model

## Notation

The following notation will be used for the autoregressive model:
- $t$ = number of days since rollout (i.e. $t_0$)
- $t_0$ = rollout, i.e. the last day before the vaccine was available
- $c_t^{obs}$ = observed cumulative uptake on day $t$
- $\sigma_t^{obs}$ = empirical estimate of the standard deviation of the observed cumulative uptake on day $t$
- $c_t$ = latent true cumulative uptake on day $t$
- $u_t$ = latent true incident uptake between days $t-1$ and $t$, i.e. $c_t - c_{t-1}$
- $G$ = grouping factors (e.g. season, geographic area, age group, race/ethnicity), indexed by $i$ with $I$ total factors

## Summary

At a high level, the autoregressive (AR) model is structured as follows:

```math
\begin{align*}
&c_{t,G}^{obs} \sim \text{Pr}(c_{t,G}^{obs}~|~c_{t,G},~\sigma_G^{obs}) \\
&c_{t,G} := \sum_{\tau=t_0}^{t} u_{\tau,G} \\
&u_{t,G} := f_{\text{AR}}(u_{t,G},~t,~\phi_G) \\
&\phi_G \sim \text{Pr}(\phi_G~|~\xi) \\
&\xi \sim \text{Pr}(\xi) \\
&t_0 \sim \text{Pr}(t_0) \\
\end{align*}
```

More details on the AR model are given below.

## Observation Layer

```math
\begin{align*}
&c_{t,G_1,...,G_I}^{obs} \sim \text{TruncNorm}(\text{location = }c_{t,G_1,...,G_I}, \text{ scale = }\sigma_{t,G_1,...,G_I}^{obs}, \text{ lower = }0, \text{ upper = }1) \\
\end{align*}
```

## Functional Structure

```math
\begin{align*}
&c_{t,G_1,...,G_I} = \sum_{\tau=t_0}^{t} u_{\tau,G_1,...,G_I} \\
&u_{t,G_1,...,G_I} = \alpha_{G_1,...,G_I} + \beta_{G_1,...,G_I} \cdot u_{t-1,G_1,...,G_I} + \gamma_{G_1,...,G_I} \cdot t + \theta_{G_1,...,G_I} \cdot u_{t-1,G_1,...,G_I} \cdot t ~~~ \text{ for }t > t_0 \\
&u_{t_0,G_1,...,G_I} = 0 \\
\end{align*}
```

## Hierarchical Structure

```math
\begin{align*}
&\alpha_{G_1,...,G_I} = \alpha + \alpha_{G_1} + ... + \alpha_{G_I} \\
&\frac{\alpha_{G_i}}{\sigma_{\alpha_{G_i}}} \sim \text{Normal}(\text{location = }0, \text{ scale = }1) ~\forall~i~\text{ in } 1, ..., I \\
\end{align*}
```

and similarly for $\beta$, $\gamma$, and $\theta$.

## Priors

```math
\begin{align*}
&\alpha \sim \text{Normal}(\text{location = }0, \text{ scale = }0.1) \\
&\sigma_{\alpha_{G_i}} \sim \text{Exponential}(\text{mean = }0.1) ~\forall~i~\text{ in } 1, ..., I \\
\end{align*}
```

and similarly for $\beta$, $\gamma$, and $\theta$. Additionally:

```math
\begin{align*}
&t_0 \sim \text{DiscreteUniform}(\text{lower = Early Date, upper = Late Date}) \\
\end{align*}
```

# Hill/Linear Model

## Notation

The following notation will be used for the Hill/linear model:
- $t$ = time since rollout, expressed as the fraction of a year elapsed since the start of the season
- $V_t^{obs}$ = number of people surveyed at time $t$ who are vaccinated
- $N_t^{obs}$ = total number of people surveyed at time $t$
- $c_t$ = latent true cumulative uptake on day $t$
- $G$ = grouping factors (e.g. season, geographic area, age group, race/ethnicity), indexed by $i$ with $I$ total factors

## Summary

At a high level, the Hill model is structured as follows:

```math
\begin{align*}
&V_{t,G}^{obs} \sim \text{Pr}(V_{t,G}^{obs}~|~c_{t,G},~N_{t,G}^{obs}) \\
&c_{t,G} := f_{\text{Hill, Linear}}(t,~\phi_G) \\
&\phi_G \sim \text{Pr}(\phi_G~|~\xi) \\
&\xi \sim \text{Pr}(\xi) \\
\end{align*}
```

Here, $t$ is rescaled by dividing by 365, so that $t$ represents the proportion of a season elapsed. Additionally, $V_{t,G}^{obs}$ and $N_{t,G}^{obs}$ are inferred from $c_{t,G}^{obs}$ and its reported 95% confidence interval, by assuming the latter is a Wald interval representing $1.96$ standard errors of the mean in each direction from $c_{t,G}^{obs}$.

## Observation Layer

```math
\begin{align*}
&V_{t,G_1,...,G_I}^{obs} \sim \text{BetaBinomial}(\text{shape1 = }\alpha_{t,G_1,...,G_I}, \text{ shape2 = }\beta, \text{ N = }N_{t,G_1,...,G_I}^{obs}) \\
\end{align*}
```

## Functional Structure

```math
\begin{align*}
&c_{t,G_1,...,G_I} = \frac{A_{G_1,...,G_I} \cdot t^{n}}{H^{n} + t^{n}} + M_{G_1,...,G_I} \cdot t \\
&\alpha_{t,G_1,...,G_I} = \frac{c_{t,G_1,...,G_I}}{1 - c_{t,G_1,...,G_I}} \cdot \beta \\
\end{align*}
```

Note that it is necessary to convert the latent true uptake into the first shape parameter of a beta distribution using the formula for the mean of a beta distribution.

## Hierarchical Structure

```math
\begin{align*}
&A_{G_1,...,G_I} = A + A_{G_1} + ... + A_{G_I} \\
&\frac{A_{G_i}}{\sigma_{A_{G_i}}} \sim \text{Normal}(\text{location = }0, \text{ scale = }1) ~\forall~i~\text{ in } 1, ..., I \\
\end{align*}
```

and similarly for $M$.

## Priors

```math
\begin{align*}
&A \sim \text{Beta}(\text{shape1 = }100.0, \text{ shape2 = }140.0) \\
&\sigma_{A_{G_i}} \sim \text{Exponential}(\text{rate = }40.0) ~\forall~i~\text{ in } 1, ..., I \\
&H \sim \text{Beta}(\text{shape1 = }100.0, \text{ shape2 = }225.0) \\
&n \sim \text{Gamma}(\text{shape = }20.0, \text{ rate = }5.0) \\
&M \sim \text{Gamma}(\text{shape = }1.0, \text{ rate = }0.1) \\
&\sigma_{M_{G_i}} \sim \text{Exponential}(\text{rate = }40.0) ~\forall~i~\text{ in } 1, ..., I \\
&\beta \sim \text{Gamma}(\text{shape = }5.0, \text{ rate = }0.05) \\
\end{align*}
```
