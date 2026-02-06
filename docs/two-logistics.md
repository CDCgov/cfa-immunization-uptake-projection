# Two logistic model

- The notation and observation model ($x_{gt}$ and $D$) are the same as in LPL.
- Here, $v_g(t)$ is a sum of two sigmoid functions. In incident space, they are two truncated normal distributions.
- Total uptake is controlled by a single random variable $A_g$
- Effects are kept on a linear scale for now; they should probably be transformed to work on a logit scale

## Model equations

```math
\begin{align*}
v_g(t) &= C_{1g} F_\mathrm{TN}(t; \tau_1, \sigma_1) + C_{2g} F_\mathrm{TN}(t; \tau_2, \sigma_2) \\
\tau_1 &= \mu_\tau - \tfrac{1}{2} \Delta_\tau \\
\tau_2 &= \mu_\tau + \tfrac{1}{2} \Delta_\tau \\
\mu_\tau &\sim \text{Beta}(100.0, 225.0) \\
\Delta_\tau &\sim \text{Exp}(25.0) \\
\sigma_i &\sim \mathrm{Exp}(50.0) \\
C_{1g} &= A_g B \\
C_{2g} &= A_g (1 - B) \\
A_g &= \mu_A + \sum_j \delta_{Aj z_{gj}} \\
\mu_A &\sim \text{Beta}(1, 1) \\
\delta_{jk} &\sim \mathcal{N}(0, \rho_j) \\
\rho_j &\sim \text{Exp}(40.0) \\
B &\sim \mathrm{Beta}(\tfrac{1}{2}, \tfrac{1}{2}) \\
\end{align*}
```
