import forestci as fci
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.ensemble._forest import (
    BaseForest,
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)
from sklearn.model_selection import train_test_split


#### variance function from fci source code ####
def calc_inbag(n_samples, forest):
    """
    Derive samples used to create trees in scikit-learn RandomForest objects.

    Recovers the samples in each tree from the random state of that tree using
    :func:`forest._generate_sample_indices`.

    Parameters
    ----------
    n_samples : int
        The number of samples used to fit the scikit-learn RandomForest object.

    forest : RandomForest
        Regressor or Classifier object that is already fit by scikit-learn.

    Returns
    -------
    Array that records how many times a data point was placed in a tree.
    Columns are individual trees. Rows are the number of times a sample was
    used in a tree.
    """

    if not forest.bootstrap:
        e_s = "Cannot calculate the inbag from a forest that has bootstrap=False"
        raise ValueError(e_s)

    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    if isinstance(forest, BaseForest):
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, forest.max_samples)

        for t_idx in range(n_trees):
            sample_idx.append(
                _generate_sample_indices(
                    forest.estimators_[t_idx].random_state,
                    n_samples,
                    n_samples_bootstrap,
                )
            )
            inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    elif isinstance(forest, BaseBagging):
        for t_idx, estimator_sample in enumerate(forest.estimators_samples_):
            sample_idx.append(estimator_sample)
            inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)

    return inbag


def simple_v_ij(
    inbag,
    pred_centered,
    n_trees,
):
    return np.sum((np.dot(inbag - 1, pred_centered.T) / n_trees) ** 2, 0)


def simple_v_ij_numerator(
    inbag,
    pred_centered,
):
    return np.sum(np.dot(inbag - 1, pred_centered.T), 0)


def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    """
    Helper functions that implements bias correction

    Parameters
    ----------
    V_IJ : ndarray
        Intermediate result in the computation.

    inbag : ndarray
        The inbag matrix that fit the data. If set to `None` (default) it
        will be inferred from the forest. However, this only works for trees
        for which bootstrapping was set to `True`. That is, if sampling was
        done with replacement. Otherwise, users need to provide their own
        inbag matrix.

    pred_centered : ndarray
        Centered predictions that are an intermediate result in the
        computation.

    n_trees : int
        The number of trees in the forest object.
    """
    n_train_samples = inbag.shape[0]
    n_var = np.mean(
        np.square(inbag[0:n_trees]).mean(axis=1).T.view()
        - np.square(inbag[0:n_trees].mean(axis=1)).T.view()
    )

    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var / n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def _centered_prediction_forest(forest, X_test, y_output=None):
    """
    Center the tree predictions by the mean prediction (forest)

    The centering is done for all provided test samples.
    This function allows unit testing for internal correctness.

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    X_test : ndarray
        An array with shape (n_test_sample, n_features). The design matrix
        for testing data

    Returns
    -------
    pred_centered : ndarray
        An array with shape (n_test_sample, n_estimators).
        The predictions of each single tree centered by the
        mean prediction (i.e. the prediction of the forest)

    """
    # In case the user provided a (n_features)-shaped array for a single sample
    #  shape it as (1, n_features)
    # NOTE: a single-feature set of samples needs to be provided with shape
    #       (n_samples, 1) or it will be wrongly interpreted!
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, -1)

    pred = np.array([tree.predict(X_test) for tree in forest])
    if "n_outputs_" in dir(forest) and forest.n_outputs_ > 1:
        pred = pred[:, :, y_output]

    pred_mean = np.mean(pred, 0)

    return (pred - pred_mean).T


def wrapper(rf, X_train, X_test):
    """
    Input different number of trees, observe how the numerator changes
    """
    inbag = calc_inbag(X_train.shape[0], rf)
    pred_diff = _centered_prediction_forest(rf, X_test)

    V_IJ_numerator = simple_v_ij_numerator(inbag, pred_diff)
    V_IJ = simple_v_ij(inbag, pred_diff, rf.n_estimators)

    return V_IJ_numerator, V_IJ


### toy data ###
np.random.seed(42)

n = 50

x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(5, 2, n)
x3 = np.random.uniform(-3, 3, n)

error = np.random.normal(0, 1, n)
y = 3.5 * x1 - 2.0 * x2 + 1.2 * x3 + 5 + error

data = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

#### RF fit ####
X = data.select(["x1", "x2", "x3"]).to_numpy()
y = data["y"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 50, random_state=123
)

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)

### manually calculate V_UIJ ###
pred_diff = _centered_prediction_forest(rf, X_test)
inbag = calc_inbag(X_train.shape[0], rf)

V_ij = simple_v_ij(inbag, pred_diff, rf.n_estimators)
V_UIJ = _bias_correction(V_ij, inbag, pred_diff, rf.n_estimators)

### V_UIJ from forestci ###
var_ci = fci.random_forest_error(rf, X_train.shape, X_test)

### Two variance should be the same ###
np.all(np.abs(V_UIJ - var_ci) < 1e-5)

### How V_ij changes with B ###
numerator_list = []
v_ij_list = []
B_list = []

for expo in np.arange(1, 6):
    B = 10**expo
    print(B)
    rf = RandomForestRegressor(n_estimators=B)
    rf.fit(X_train, y_train)
    numerator, v_ij = wrapper(rf, X_train, X_test)
    numerator_list.append(numerator)
    B_list.append(B)
    v_ij_list.append(v_ij)

numerators = np.array(numerator_list)
v_ijs = np.array(v_ij_list)
Bs = np.array(B_list)

plt.scatter(np.log(Bs), v_ijs)
plt.scatter(np.log(Bs), np.log(numerators))
