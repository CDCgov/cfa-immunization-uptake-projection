# Random Forest

## Definition

Random forest is an ensemble learning method to solve classification or regression problem, by creating multiple decision trees during training. A random forest is a series of boostrap aggregated decision trees.

## Desicion tree

A decision tree trains the data from recursively splitting the data into smaller subsets based on the values of features (predictors) and choosing splits that can best separate the response variables. It aims to create a tree structure that can make accurate predictions. It can be applied to classification or regression problem. Here we focus on regression problem.

### Decision tree for regression

Decision tree recursively splits the data and creates leaf nodes (branches) to reduce mean squared error (MSE). The splitting will stop when:

- Maximum depth (defined by user) is reached.
- A minimum number of samples per node is reached.
- All samples belong to one grouping class (the node becomes pure).

To have a split, decision tree will calculate the resulting MSE given a threshold for a certain predictor to check if the MSE will be reduced, for all the possible thresholds from all the predictors, and select the threshold of the predictor that can mostly reduce MSE. Note that, this cannot guarantee the global optimal to be found, because each node only explores their local optimum.

### Issues

The selection of nodes is sensitive to small change of the data, which often leads to overfitting to the dataset. To avoid overfitting and instead capturing the true relationship between the response variable and predictors, we need to use random forest.

## Random forest

Random forest used a technique called "boostrap aggregating" or "bagging", which will repeatedly select a random sample with replacement from the training dataset and fits multiple trees to these samples. In this way, multiple trees with different structures are trained. The final prediction for the response variable in the testing data will be the average of predictions from these trees, given predictors in the testing data.

Writing this mathematically:

Given a training set $X = x_1, ..x_n$ with response $Y = y_1,..y_n$, if bagging $B$ times:

For $b = 1, ..., B$:
1. Sample, with replacement, $n$ training examples from $X, Y$; call these $X_b, Y_b$.
2. Train a regression tree $f_b$ on $X_b, Y_b$.

After training, predictions for the predictors $x'$ in testing data can be made by averaging the predictions from all the individual regression trees on $x'$:

```math
\hat f = \frac{1}{B}\sum_{b=1}^B f_b(x')

```
The prediction error (as variance) can also be derived:

```math
\begin{align}

V = \frac{\sum_{b=1}^B(f_b(x') -\hat f)^2}{B-1}

\end{align}
```
While based on [Wager et al](https://jmlr.org/papers/volume15/wager14a/wager14a.pdf), estimating the variance of prediction error (Eq(1)) can be challenging, as it includes two sources of noises: sampling noise from data collection, and Monte Carlo noise from the boostrap replicates. Specifically, Monte Carlo noise will inflate when $B$ is small. This study proposes a bias-corrected estimator of variance for random forest. It is:

```math

V = \sum_{i=1}^n \frac{\sum_b (N_{bi}^* -1)(f_b(x)-\hat f)}{B} - \frac{n}{B^2}\sum_{b=1}^B(f_b(x) - \hat f)^2

```
where $n$ is number of the data points, $N_{bi}^*$ is the number of times the $i^{th}$ observation appears in the boostrap sample $b$, $f_b(x)$ is the prediction at $x$ from tree $b$, $\hat f$ is the average prediction across all the trees at $x$. This estimator is used by R package `randomForestCI` and is inherited by Python package `forestci`.

Based on central limit theorem, the distribution of $\hat f$ is normal, then we can calculate the 95% confidence interval as $[\hat f - 1.96 \sqrt(V), \hat f + 1.96 \sqrt(V)]$.
