# Draft analytical plan

## Definitions

Every data point has an associated _reference date_, when the measured physical process occurred, and a _report date_, when that data became available for forecasting. For example, if immunization uptake was 50% on April 1, and that information became available on April 5, the reference date is April 1 and the report date is April 5.

A _forecast_ is:

- a model, including its parameters and training data,
- a _forecast date_,
- a set of reference dates after the future relative to the forecast date, and
- the predicted values for each reference date.

In _prospective_ forecasting, the forecast date is the moment that the forecast was actually run. The training data is all the data that is available as of that moment, and the predictions are for reference dates after the forecast date.

In _retrospective_ forecasting, the forecast date is a date in the past (relative to today). The training data is all the data that was available as of the forecast date, that is, all the data that has a report date before the forecast date. Predictions are for reference dates after the forecast date but before today.

The forecast _horizon_ is the difference between the forecast date and a forecast reference date. A single forecast can cover multiple horizons. For example, a forecast with a forecast date of October 1 might include forecasts for the reference dates October 14, October 21, and so forth.

A _score function_ takes predicted values produced by a single model (usually at some specific horizon), compares them to the true values, and returns a _score_ that quantifies the predictive performance of that model.

## Analytical plan

Perform retrospective forecasting to compare model performance:

- Define the _start of season_ as September 1(?) and _end of season_ as April 1(?) or similar, TBD
- For the 3(?) most recent seasons, let the forecast dates be every Saturday from the start of the season to the end of the season
- For each model, forecast season, and forecast date: predict uptake for each week from the forecast date to the end of the season
  - Note that this includes training data from prior seasons as well as that season up to the forecast date
  - Note that earlier forecasts will produce more predictions (and horizons) than later forecasts
- For each model, score function, and horizon: generate a score
  - For the end-of-season uptake score function(s), score(s) will evaluate predictions for a single reference date (for each horizon)
  - For other score functions, scores will include multiple reference dates for each horizon
- Compare the scores across models (stratifying by horizon) and within models (across horizons)
