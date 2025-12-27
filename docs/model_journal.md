# Model "journal"

Choosing a model structure has been trickier than expected. This is an informal record of the attempts that have been made, their outcomes, and consequent directions.

## Cumulative S Curves

The Scenarios team would like to use vaccine uptake forecasts as input for their ODE models. For this purpose, they need uptake curves that are continuously differentiable, not forecasts that are a series of point estimates. Autoregressive and stochastic models are not suitable for this purpose. Consequently, families of S-curves that directly model cumulative uptake will be prioritized, starting with the Hill function.

## Hill Function

In brief, the original Hill function model considers latent true uptake to follow a Hill curve shape. The final uptake and midpoint parameters ("A" and "H", respectively) can deviate additively from their overall averages based on grouping factors (e.g. season, state), while the steepness parameter ("n") only has one overall value. Observed uptake is centered on the latent true uptake but may have error in either direction. The magnitude of this error is derived from the 95% confidence interval reported along with each point estimate in the NIS data.

### Trouble Pt. 1: Truncated Normal Observation

The Hill model would not fit - the MCMC chains remained stationary and returned hundreds of identical draws with ESS = 1.0. This happened whether grouping factors were included or not (i.e. one curve fit across all seasons at the national scale). The stationary chains were solved by ignoring the empirical estimates of observation error: when observation error was fixed at a generous value (0.03) or was fit as a free parameter, the MCMC chais were no longer stationary.

Why did empirical observation error break MCMC? The original Hill model used a truncated Normal draw to describe the observation process. The reported 95% confidence intervals were assumed to be Wald intervals, such that an interval's half-width divided by 1.96 approximates the standard deviation of the truncated Normal. These standard deviations were often on the order of 0.001, implying that the observed uptake curves are very close to the latent true uptake. But the Hill function does not fit the data that well: especially in the latter half of seasons, true uptake continues creeping upward while the Hill function asymptotes. Thus, no parameter set exists that can get the Hill function close enough to all data points, and MCMC gets stuck in flat portions of the likelihood landscape.

### Solution Pt. 1: Beta-Binomial Observation

MCMC chains were unstuck by reinterpretting the empirical confidence intervals in terms of the actual data collection process. Cumulative uptake is estimated by the proportion p of N phone survey participants who report being vaccinated. By considering an interval's half-width divided by 1.96 to be the standard error of the mean (SEM) for the reported uptake proportion p, N was estimated at each data point. Sensibly, stimated N is on the order of 1,000 for individual states and 50,000 at the national scale.

With estimates of pN and N in hand, the observation process was replaced with a beta-binomial likelihood, which inherently permits observations to vary farther from the latent true uptake, compared to the truncated Normal likelihood. Consequently, the MCMC chains began sampling parameter space more freely.

### Trouble Pt. 2: Hill Function Shape

Even with MCMC proceeding, other warning signs arose:

- When grouping on season alone, season-specific deviations in both A and H from their overall averages have very wide 95% credible intervals, straddling 0. And yet, it is clear that uptake curves do differ from one another across seasons.
- When grouping on season and state, the fitting proceeds very slowly (1-2 it/s). A, A-deviations-by-season, H, and H-deviations-by-season all had very low ESS (40-60, despite 500 samples after warmup). A-deviations-by-state had even lower ESS (10-15). H-deviations-by-state had higher ESS, but the magnitude in variation in H among states was estimated very close to 0.

Together, these warning signs suggest some non-identifiability among the parameters that vary by grouping factor, perhaps again driven by the poor fit of the Hill function to uptake curves.

### Solution Pt 2: Logistic + Linear Functions

Many warning signs were alleviated by changing the structure of the latent true uptake from a pure Hill function to a logistic function plus a slope-only linear function (intercept = 0). In this model, the linear slope "M" and the logistic asymptote "A" can deviate additively from their overall averages by group, while the logistic midpoint "H" and steepness "n" are fixed across groups. In particular, this mixed function allows uptake to continue creeping upward late in a season.

When grouping on season and state, the fitting proceeds faster. A and A-deviations-by-state still have low ESS (50-100, with 500 samples after warmup). A-deviations-by-season, M, and M-deviations-by-state/season all have good ESS (100-500+). The magnitudes of A- and M-deviations are confidently estimated greater than 0. The 95% credible intervals for A- and M-deviations from the overall average often do not overlap 0, for many states and seasons. There are 0 divergences.

The more promising aspects of the MCMC summary likely reflect a more appropriate fit of the latent true uptake model to the shape of the data.
