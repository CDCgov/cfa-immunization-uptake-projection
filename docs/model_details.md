# Overview

These are the mathematical details of the models used to capture and forecast vaccine uptake. There are currently just one model: a mixture of a logistic and linear function. This model proposes a latent true uptake curve, which is subject to observation error. A hierarchy accounts for the unique effects of grouping factors (e.g. season, geography, age) on model parameters.

# Logistic + Linear (LL) Model

## Notation

The following notation will be used for the LL model:
- $t$ = time since the start of the season, expressed as the fraction of a year elapsed
- $V_t^{obs}$ = number of people surveyed at time $t$ who are vaccinated
- $N_t^{obs}$ = total number of people surveyed at time $t$
- $c_t$ = latent true cumulative uptake on day $t$
- $G$ = grouping factors (e.g. season, geographic area, age group, race/ethnicity), indexed by $i$ with $I$ total factors

## Summary

At a high level, the LL model is structured as follows:

```math
\begin{align*}
&V_{t,G}^{obs} \sim \text{Pr}(V_{t,G}^{obs}~|~c_{t,G},~N_{t,G}^{obs}) \\
&c_{t,G} := f_{\text{Hill, Linear}}(t,~\phi_G) \\
&\phi_G \sim \text{Pr}(\phi_G~|~\xi) \\
&\xi \sim \text{Pr}(\xi) \\
\end{align*}
```

Here, $t$ is rescaled by dividing by 365, so that $t$ represents the proportion of a season elapsed. Additionally, $V_{t,G}^{obs}$ and $N_{t,G}^{obs}$ are inferred from $c_{t,G}^{obs}$ and its reported 95% confidence interval, by assuming the latter is a Wald interval representing $1.96$ standard errors of the mean in each direction from $c_{t,G}^{obs}$. As a result, the standard error of the mean $\sigma_{t,G}^{SEM}$ is considered known for each data point, and $V_{t,G}^{obs}$ and $N_{t,G}^{obs}$ are as follows:

```math
\begin{align*}
&N_{t,G}^{obs} = \frac{c_{t,G}^{obs} \cdot (1-c_{t,G}^{obs})}{{\sigma_{t,G}^{SEM}}^2} \\
&V_{t,G}^{obs} = N_{t,G}^{obs} \cdot c_{t,G}^{obs} \\
\end{align*}
```

## Observation Layer

The observed uptake is considered a draw from the beta-binomial distribution, governed in part by the true latent uptake in the population.

```math
\begin{align*}
&V_{t,G_1,...,G_I}^{obs} \sim \text{BetaBinomial}(\text{shape1 = }\alpha_{t,G_1,...,G_I}, \text{ shape2 = }\beta_{t,G_1,...,G_I}, \text{ N = }N_{t,G_1,...,G_I}^{obs}) \\
\end{align*}
```

Note that the shape parameters $\alpha$ and $\beta$ are not declared explicitly. Rather they are implied by an alternate mean and concentration parametrization, described below.

## Functional Structure

The model's functional structure describes the latent true uptake curve:

```math
\begin{align*}
&c_{t,G_1,...,G_I} = \frac{A_{G_1,...,G_I}}{1 + e^{-n \cdot (t - H)}} + M_{G_1,...,G_I} \cdot t \\
\end{align*}
```

 $c_{t,G_1,...,G_I}$ serves as the mean of the beta distribution in the beta-binomial likelihood in the observation-layer. A fixed concentration parameter $d$ is also required. From the mean and concentration, the two shape parameters of the beta distribution are as follows:

```math
\begin{align*}
&\alpha_{t,G_1,...,G_I} = c_{t,G_1,...,G_I} \cdot d \\
&\beta_{t,G_1,...,G_I} = (1 - c_{t,G_1,...,G_I}) \cdot d \\
\end{align*}
```

## Hierarchical Structure

Certain parameters of the latent true uptake curve have group-specific deviations, determined as follows:

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
&A \sim \text{Beta}(\text{shape1 = }100.0, \text{ shape2 = }180.0) \\
&\sigma_{A_{G_i}} \sim \text{Exponential}(\text{rate = }40.0) ~\forall~i~\text{ in } 1, ..., I \\
&H \sim \text{Beta}(\text{shape1 = }100.0, \text{ shape2 = }225.0) \\
&n \sim \text{Gamma}(\text{shape = }25.0, \text{ rate = }1.0) \\
&M \sim \text{Gamma}(\text{shape = }1.0, \text{ rate = }10.0) \\
&\sigma_{M_{G_i}} \sim \text{Exponential}(\text{rate = }40.0) ~\forall~i~\text{ in } 1, ..., I \\
&d \sim \text{Gamma}(\text{shape = }350.0, \text{ rate = }1.0) \\
\end{align*}
```
