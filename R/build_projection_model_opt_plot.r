#' Model future uptake from current uptake
#'
#' BEWARE: THIS DOES NOT YET ACCOUNT FOR MULTI-YEAR OR MULTI-STATE DATA SETS
#'
#' Future incident uptake can be predicted from current incident uptake.
#' A model can be built and fit to project incident uptake into the future.
#'
#' The default model says that current incident uptake is a linear function
#' of the previous incident uptake, time elapsed since rollout, and their
#' interaction.
#'
#' All modeling is performed on the daily-average incident uptake to
#' remove accumulation dependence among data points and to
#' account for slight variations in incident reporting intervals.
#' The first data point may be dropped before fitting the model
#' if there is a long lag between the vaccine rollout date and
#' the date that data collection began.
#' All variables are standardized prior to fitting the model.
#'
#' UPDATES FOR THE FUTURE:
#' - Model must incorporate year-to-year variability
#' - Model must incorporate state-to-state variability
#' - A variety of models should be tried and competed, or formula provided
#'
#' @description Fit a model to project uptake into the future
#'
#' @param data uptake data to fit the projection model
#' @param drop_first whether to drop the first data point
#'
#' @return a brms model object
#'
#' @export
#'
build_projection_model2 <- function(data, drop_first = NULL, output = F) {
  if (is.null(drop_first)) {
    intervals <- diff(data$elapsed)
    if (abs(data$elapsed[1] - mean(intervals)) > (sd(intervals) + 1)) {
      drop_first <- TRUE
    } else {
      drop_first <- FALSE
      intervals <- c(data$elapsed[1], intervals)
    }
  }

  if (drop_first) {
    data <- data[-1, ]
  }

  model <- brms::brm(daily_std ~ previous_std * elapsed_std,
    data = data,
    family = gaussian()
  )

  if (output) {
    print(summary(model))
    plot(model)
  }

  return(model)
}
