import forestci as fci
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


### RF V_UIJ error ###
def get_rf_pred_vuij(B, X_train, X_test, y_train):
    rf = RandomForestRegressor(n_estimators=B, random_state=123)
    rf.fit(X_train, y_train)
    var_ci = fci.random_forest_error(rf, X_train.shape, X_test)

    y_pred = rf.predict(X_test)
    lci = y_pred - 1.96 * np.sqrt(var_ci)
    uci = y_pred + 1.96 * np.sqrt(var_ci)

    pred_df = pl.DataFrame({"pred": y_pred, "lci": lci, "uci": uci})

    return pred_df


### RF OOB error ###
def get_rf_pred_oob(B, X_train, X_test, y_train, alpha=0.05):
    rf = RandomForestRegressor(n_estimators=B, random_state=123, oob_score=True)
    rf.fit(X_train, y_train)

    resid = y_train - rf.oob_prediction_

    y_pred = rf.predict(X_test)
    lci = y_pred + np.quantile(resid, alpha / 2)
    uci = y_pred + np.quantile(resid, 1 - alpha / 2)

    pred_df = pl.DataFrame({"pred": y_pred, "lci": lci, "uci": uci})

    return pred_df


### LM ###
def get_lm_pred(X_train, X_test, y_train, alpha=0.05):
    lr = sm.OLS(y_train, sm.add_constant(X_train)).fit()

    y_pred = lr.get_prediction(sm.add_constant(X_test)).summary_frame(alpha=alpha)

    pred_mean = y_pred["mean"]
    pred_lci = y_pred["obs_ci_lower"]
    pred_uci = y_pred["obs_ci_upper"]

    pred_df = pl.DataFrame({"pred": pred_mean, "lci": pred_lci, "uci": pred_uci})

    return pred_df


### Toy data ###
np.random.seed(42)

n = 50

x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(5, 2, n)
x3 = np.random.uniform(-3, 3, n)

error = np.random.normal(0, 2, n)
y = 3.5 * x1 - 2.0 * x2 + 1.2 * x3 + 5 + error

data = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

X = data.select(["x1", "x2", "x3"]).to_numpy()
y = data["y"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

### linear regression ###
lm_df = get_lm_pred(X_train, X_test, y_train)

### Random forest ###
# number of trees to try #
Bs = [100, 1000, 5000, 10000]

rf_vuij_list = [get_rf_pred_vuij(B, X_train, X_test, y_train) for B in Bs]
rf_oob_list = [get_rf_pred_oob(B, X_train, X_test, y_train) for B in Bs]

## plot lm vs rf with varying number of trees ##
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()

titles = (
    ["linear regression"]
    + [f"RF_V_UIJ {B} trees" for B in Bs]
    + [f"RF_OOB {B} trees" for B in Bs]
)
x_coords = np.arange(len(y_test))
all_df = [lm_df] + rf_vuij_list + rf_oob_list

for i, ax in enumerate(axes):
    ax.scatter(x_coords, y_test, color="grey")
    ax.plot(all_df[i]["pred"])
    ax.fill_between(x_coords, all_df[i]["lci"], all_df[i]["uci"], alpha=0.4)
    ax.set_title(titles[i])

plt.tight_layout()
plt.savefig("output/demo_rf/toy_data.png")
