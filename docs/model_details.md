# Model details

These are the mathematical details of the models used to capture and forecast vaccine uptake. There are currently just one model: a mixture of a logistic and linear function. This model proposes a latent true uptake curve, which is subject to observation error. A hierarchy accounts for the unique effects of grouping factors (e.g. season, geography, age) on model parameters.

## Logistic Plus Linear (LPL) Model

### Notation

The following notation will be used for the LPL model:

- $t$: Time, where $t=0$ is the start of the season, measured in years (i.e., $t=1$ is 1 year after $t=0$)
- $v_{gt}$: observed uptake among group $g$ at time $t$
- $V_g(t)$: latent true cumulative uptake on day $t$
- $g = (\mathcal{X}_{g1}, \ldots, \mathcal{X}_{gN})$: each group $g$ encodes the values $\mathcal{X}_{gi}$ of each of the $N$ modeled features $i$. In our main analysis, the features are season, geographic area, and age group.

### Formulation

The observed uptakes are beta distributed around the latent uptake:

```math
v_{gt} \sim \mathrm{Beta}(\alpha_{gt}, \beta_{gt})
```

The values $\alpha_{gt}$ and $\beta_{gt}$ are chosen so that the mean of this distribution is $V_g(t)$ and the variance best matches the reported NIS confidence intervals (see [below](#beta-variance)).

The latent uptake follows the LPL:

$$
V_g(t) = \frac{A_g}{1 + \exp\left\{-K (t-\tau)\right\}} + M_g t
$$

The group-level amplitude $A_g$ and slope $M_g$ are sums of feature value-level deviations from a grand mean. For $A_g$:

$$
A_g = \mu_A + \sum_{i=1}^N \delta_{A,i,\mathcal{X}_{gi}}
$$

where $\delta_{Aix}$ is the deviation for parameter $A$, feature $i$ (e.g., season), and group $g$ (which has, say, season $\mathcal{X}_{gi}$).

We specify a prior for each grand mean like $\mu_A$. The deviations are assumed to be normally distributed:

$$
\delta_{Aix} \sim \mathcal{N}\left( 0, \sigma^2_A \right)
$$

and we specify a prior for each standard deviation $\sigma_A$.

The other logistic parameters $K$ and $\tau$ are assumed common to all groups.

### Priors

```math
\begin{align*}
\mu_A &\sim \text{Beta}(100.0, 180.0) \\
\sigma_A &\sim \text{Exponential}(40.0) \\
\tau &\sim \text{Beta}(100.0, 225.0) \\
K &\sim \text{Gamma}(\text{shape} = 25.0, \text{rate} = 1.0) \\
\mu_M &\sim \text{Gamma}(\text{shape} = 1.0, \text{rate} = 10.0) \\
\sigma_M &\sim \text{Exponential}(40.0) \\
\end{align*}
```

### Beta variance

The NIS data report the point estimate, which we interpret as the observed value $v_{gt}$, and 95% confidence intervals. Let $q_1 = 0.025$ and $q_2 = 1-q_1$ be those quantiles and $(x_1, x_2)$ be the confidence interval. Find the $n>0$ that minimizes $\sum_{j \in \{1, 2\}} \left(F^{-1}(q_j; \alpha, \beta) - x_j\right)^2$, where $\alpha = v_{gt} n$ and $\beta = (1 - v_{gt})n$.
