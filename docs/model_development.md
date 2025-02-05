# Purpose

This is a summary of the vaccine uptake models under consideration, as well as some issues and development priorities for each.

# Basic Structures

There are two broad types of models under consideration: autoregressive (AR) and full-curve (FC) models.

- AR: Incident uptake at time $t$, $u_t$, is a function previous incident uptake value(s), among other predictors.
- FC: Cumulative uptake at time $t$, $c_t$, is a function of time, among other predictors.

Several facets of each model are discussed, including misspecification, hyperparameters, rollout dates, forecast uncertainty, and developmental priorities.

Importantly, for the first version of each model, the observed cumulative uptake $c_t$, and the incident uptake $u_t = c_t - c_{t-1}$ that it implies, are considered synonymous with true uptake. In future versions of each model, it will be important to separate latent true uptake from the observed uptake. The former is never observed and the latter is subject to error, which is estimated in the NIS data.

# Autoregressive (AR) Models

Let $u_t$ be the incident uptake at time $t$, where $t$ is measured in weeks since rollout.

A generic AR model of the I<sup>th</sup> order has the form:

$$
\begin{align*}
&u_t \sim N(\mu_t, \sigma) \\
&\mu_t = \sum_{i=0}^{t-1} \beta_{i}u_{i} \\
&\beta_{i},~\sigma \sim \text{prior distributions} \\
\end{align*}
$$

The "linear incident uptake model" currently implemented is a first-order AR model, but with the added complexity of absolute time as a predictor, too:

$$
\begin{align*}
&u_t \sim N(\mu_t, \sigma) \\
&\mu_t = \alpha + \beta_{u}u_{t-1} + \beta_{t}t + \beta_{tu}tu_{t-1} \\
&\alpha,~\beta_x,~\sigma \sim \text{prior distributions} \\
\end{align*}
$$

Essentially, this model says that the uptake rate changes through time ($\beta_t$) with some inertia ($\beta_u$), and that the inertia itself can change through time ($\beta_{tu}$).

## Misspecification

Incident uptake $u_t$ is naturally bounded by $[0,~1-c_{t-1}]$, meaning that over the course of a week $t$, the number of people who get vaccinated cannot be fewer than 0 or exceed the number of people who were unvaccinated in week $t-1$

However, the distribution $N(\mu, \sigma)$ is unbounded, meaning that the model is free to predict $u_t$ outside its natural bounds. In fact, even the mean of this distribution $\mu$ could lie outside these natural bounds: because the parameters $\alpha$ and $\beta_x$ are currently assigned Gaussian priors, they are also unbounded and could take on extreme values. All together, this is misspecification.

Practically, nonsensical predictions as a result of this misspecification can be avoided by choosing priors carefully. In particular,

- Priors for $\alpha$ and $\beta_x$ should have a zero mean and small variance.
- The prior for $\sigma$ should have a small expectation.

Under these conditions, parameter combinations that predict nonsensical future incident uptake are unlikely to be drawn from the posterior. Nonetheless, refactoring the linear incident uptake model to avoid misspecification altogether is desirable.

The best approach may involve separating latent true uptake (cumulative $\hat{c}_t$ and incident $\hat{u}_t$) from observed uptake (cumulative $c_t$ and incident $u_t$), and making use of the observed variability ($\sigma_t$) that is also reported alongside $c_t$ in the data. This allows the natural limits of $[0,~1]$ to be imposed on the observed uptake $c_t$, via a truncated normal distribution, denoted $TruncNorm(\text{mean, variance, lower bound, upper bound})$.

```math
\begin{align*}
&c_t \sim TruncNorm(\hat{c}_t, \sigma_t, 0, 1) \\
&\hat{c}_t = \sum_{i=0}^{t} \hat{u}_i \\
&\hat{u}_i \sim N(\mu_i, \sigma) \\
&\mu_i = \alpha + \beta_{u}\hat{u}_{i-1} + \beta_{t}t + \beta_{tu}t\hat{u}_{t-1} \\
&\alpha,~\beta_x,~\sigma \sim \text{prior distributions} \\
\end{align*}
```

Even still, $\hat{u}_i$ might escape $[0,~1-\hat{c}_i]$, so $\hat{c}_t$ might escape $[0,~1]$. The truncated normal distribution for $c_t$ prevents this from ultimately manifesting in nonsensical observed cumulative uptake, but some misspecification still lurks.

## Hyperparameters

Realistically, the $\alpha$ and $\beta_x$ parameters that control uptake rate probably differ slightly but systematically from season to season (where "season" refers to an overwinter disease season, e.g. "2024/2025"). And multiple past seasons of data are available for training models (two past seasons for covid, and many more for flu). Therefore, it would be wise to introduce hyperparameters with a subscript "s" for season:

$$
\begin{align*}
&u_t \sim N(\mu_t, \sigma) \\
&\mu_t = \alpha_s + \beta_{u,s}u_{t-1} + \beta_{t,s}t + \beta_{tu,s}tu_{t-1} \\
&\alpha_s \sim N(\alpha,~\sigma_{\alpha}) \\
&\beta_{t,s} \sim N(\beta_t,~\sigma_{\beta_t}) \\
&\beta_{u,s} \sim N(\beta_u,~\sigma_{\beta_u}) \\
&\beta_{tu,s} \sim N(\beta_{tu},~\sigma_{\beta_{tu}})  \\
&\alpha,~\beta_x,~\sigma_{\alpha},~\sigma_{\beta_x},~\sigma \sim \text{prior distributions} \\
\end{align*}
$$

Here, $\alpha$ and $\beta_x$ are hyperparameters: they govern hyperdistributions, from which a separate draw is made for each season.

Factors other than season could also be used to group the data. For example, further hyperparameters could be used to infer different parameter values (drawn from a common distribution) that describe the uptake in different geographic regions or among different demographic groups.

## Rollout Dates

The rollout date $t_0$ is the first date on which the vaccine was available. Rollout date is also considered the first data point of the season (that is, $t_0 = 0$ when indexing on time), when cumulative uptake is sure to be 0 (that is, $c_{t_0} = c_0 = 0$).

For covid, $t_0$ is known with some precision, usually around September 1. For flu, $t_0$ is known with less precision, often sometime in July.

Current models assume that $t_0$ is known and that $c_{t_0} = 0$. Future models should treat $t_0$ as a parameter to be fit, since it is often not known in reality.

## Forecast Uncertainty

Forecasting with AR models naturally produces a cone of uncertainty that expands into the future. Each draw from the posterior distribution is a unique combination of parameter values that defines a trajectory of uptake going forward (still with some stochastic influence on observations, from $\sigma$). All these trajectories sprout from the last observed data point and diverge as they move into the future.

## Priorities

The top priority for refining the linear incident uptake model is to incorporate hyperparameters to group the data by season. Then, the model fits and forecasts should be inspected for undesirable behavior. Especially if undesired behavior (likely due to misspecification) arises, refactoring the model to separate latent true uptake from observed uptake and to treat rollout as a parameter should come next.

# Full Curve (FC) Models

Let $c_t$ be the cumulative uptake at time $t$, where $t$ is measured in weeks since rollout.

Many families of sigmoid curves might reasonably capture cumulative uptake. A simple choice is the Hill function. Such a model would be written:

$$
\begin{align*}
&c_t \sim N(\mu_t, \sigma) \\
&\mu_t = \frac{A~t^n}{H^n + t^n} \\
&A,~H,~n \sim \text{prior distributions} \\
\end{align*}
$$

Essentially, this model says that uptake will asymptote to a maximum of $A$ and reach half maximal at time $t=H$, with steepness controlled by $n$.

## Misspecification

Cumulative uptake $c_t$ is naturally bounded by $[0,~1]$, meaning that the number of people who get vaccinated cannot be fewer than 0 or exceed the number of people in the population. The Hill model naturally obeys these constraints.

However, the Hill model is misspecified in another way: while $c_t$ can never be exactly 0, it must be true that uptake is 0 before the rollout date. That said, the Hill function can get arbitrarily close to 0, so this is not likely a problem in practice.

## Rollout

Just as for AR models, future FC models should also include $t_0$ as a parameter to be fit. But in this case, if the Hill function or another function that never hits 0 is used, it might be useful to define $c_{t_0} < \varepsilon$ where $\varepsilon$ is some small number, e.g. $10^{-6}$.

## Hyperparameters

Again, the $A,~H,~n$ parameters that control uptake probably differ slightly but systematically from season to season. So hyperparameters can again be introduced with a subscript "s":

$$
\begin{align*}
&c_t \sim N(\mu_t, \sigma) \\
&\mu_t = \frac{A_s~t^{n_s}}{H_s^{n_s} + t^{n_s}} \\
&A_s \sim N(A,~\sigma_A) \\
&H_s \sim N(H,~\sigma_H) \\
&n_s \sim N(n,~\sigma_n) \\
&A,~H,~n,~\sigma_A,~\sigma_H,~\sigma_n,~\sigma \sim \text{prior distributions} \\
\end{align*}
$$

And again, factors other than season, such as geographic area or demographic group, could be used to group the data.

## Forecasting Uncertainty

Sensible forecast uncertainty is the biggest challenge for FC models. Because the model parameters control the shape of the whole uptake curve (not just the value of the "next" data point, as in AR models), uncertainty in the posterior distribution means that:

- A forecast may not "pick up" exactly where the last observed data point "left off."
- A forecast may have wide uncertainty immediately after the forecast date, which does not expand much into the future.

Suppose the last observed data point is $c_{T}$. Rather than treating $c_T$ as known (as an AR model would treat $u_{T}$ as known), an FC model still predicts $c_{T}$. If the current season does not look much like the average of past seasons in the training data, the mean model prediction for $c_{T}$ will be far from the reported value of $c_{T}$.

Hyperparameters for season will help address this problem but will not solve it entirely, due to shrinkage toward the mean during partial pooling. Further thought is necessary to solve these problems.

## Priorities

The top priority for choosing and building an FC model is to explore how well the problems with forecasting uncertainty could be minimized. The effect of hyperparameters for season should be explored first, followed by a search for FC models constructed with more reasonable forecasting uncertainty in mind (e.g. hypertabastic models?). If/when a sensible FC model is identified, it must be built into the codebase.
