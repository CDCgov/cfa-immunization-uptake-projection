# Model details

These are the mathematical details of the models used to capture and forecast vaccine uptake. There are currently just one model: a mixture of a logistic and linear function. This model proposes a latent true uptake curve, which is subject to observation error. A hierarchy accounts for the unique effects of grouping factors (e.g. season, geography, age) on model parameters.

## Logistic Plus Linear (LPL) Model

### Notation

The following notation will be used for the LPL model:

- $t$: time since the start of the season, measured in $\text{year}^{-1}$
- $n_{gt}$: number of people in group $g$, surveyed at time $t$
- $x_{gt}$: number of people in group $g$, surveyed at time $t$, who are vaccinated
- $v_g(t)$: latent true coverage among group $g$ at time $t$
- $z_{gj}$: integer index indicating the level of the $j$-th feature (e.g., season, geography) for group $g$.

For example, let the features be season and geography, in that order. Let group 5 be associated with the fourth season (say, 2018/2019) and the third geography (say, Alaska). Then $z_{51} = 4$ and $z_{52} = 3$.

### Data source

- $n_{gt}$ is drawn from the `sample_size` column of the NIS data
- $x_{gt}$ is approximated as $\mathrm{round}(\hat{v}_{gt}, n_{gt})$, where $\hat{v}_{gt}$ is the `estimate` column

### Summary

For each group $g$ (e.g., season and geography), the latent coverage $v_g(t)$ is assumed to be a sum of a logistic curve (i.e., the rate incident vaccination looks like a bell curve) and a linear increase (with intercept fixed at $t=0$). The shape parameter $K$ and midpoint $\tau$ of the logistic curve are assumed to be common to all groups (including across seasons). The height $A_g$ of the logistic curve is a grand mean $\mu_A$ plus effects $\delta_{A,j,z_{gj}}$ for each feature $j$ and value $z_{gj}$ of that feature for that group. For example, the $A_g$ for Alaska in 2018/2019 will be the grand mean $\mu_A$, plus the Alaska effect, plus the 2018/2019 effect. There are no cross-terms.

The slopes $M_g$ follow a similar pattern.

The actual observations $x_{gt}$ are beta-binomial-distributed around the mean $v_g(t) \cdot n_{gt}$, with variance modified by an extra parameter $D$.

### Model equations

```math
\begin{align*}
x_{gt} &\sim \mathrm{BetaBinom}\big(v_g(t) \cdot D, [1-v_g(t)] \cdot D, n_{gt}\big) \\
v_g(t) &= \frac{A_g}{1 + \exp\{- K \cdot (t - \tau)\}} + M_g t \\
A_g &= \mu_A + \sum_j \delta_{Aj z_{gj}} \\
M_g &= \mu_M + \sum_j \delta_{Mj} \\
\mu_A &\sim \text{Beta}(100.0, 180.0) \\
\mu_M &\sim \text{Gamma}(\text{shape} = 1.0, \text{rate} = 10.0) \\
\delta_{Ajk} &\sim \mathcal{N}(0, \sigma_{Aj}) \\
\delta_{Mjk} &\sim \mathcal{N}(0, \sigma_{Mj}) \\
\sigma_{Aj} &\sim \text{Exp}(40.0) \\
\sigma_{Mj} &\sim \text{Exp}(40.0) \\
K &\sim \text{Gamma}(\text{shape} = 25.0, \text{rate} = 1.0) \\
\tau &\sim \text{Beta}(100.0, 225.0) \\
D &\sim \text{Gamma}(\text{shape} = 350.0, \text{rate} = 1.0) \\
\end{align*}
```

Note that:

$$
\begin{align*}
\mathbb{E}[x_{gt}] &= v_g(t) \cdot n_{gt} \\
\mathrm{Var}[x_{gt}] &= v_g(t) \cdot [1-v_g(t)] \cdot \frac{n_{gt} (n_{gt} + D)}{D+1}
\end{align*}
$$
