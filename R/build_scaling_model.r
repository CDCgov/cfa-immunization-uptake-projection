#' Model one source of uptake data from another
#'
#' Different sources of uptake data may systematically disagree.
#' A model can be built and fit to scale one source to the other.
#'
#' The default model says that the uptake data from the target source
#' is a linear function of the input data, time elapsed since rollout,
#' and their interaction. If multiple states are represented, each state
#' is modeled separately via group-level variation of the intercept and
#' the input data coefficients.
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
#' - A variety of models should be tried and competed, or formula provided
#'
#' @description Fit a model to scale uptake date from one source to another
#'
#' @param input uptake data to use as a predictor
#' @param target uptake data to use as the outcome
#' @param drop_first whether to drop the first data point
#'
#' @return a brms model object
#'
#' @export
#'
build_scaling_model <- function(input, target, drop_first = NULL) {

  if (is.null(drop_first)) {
    input_intervals <- c(input$elapsed[1], diff(input$elapsed))
    target_intervals <- c(target$elapsed[1], diff(target$elapsed))

    common_dates <- intersect(input$elapsed, target$elapsed)

    input_intervals <- input_intervals[input$elapsed %in% common_dates]
    target_intervals <- target_intervals[target$elapsed %in% common_dates]

    if ((input_intervals[1] - mean(input_intervals[-1])) >
          (sd(input_intervals[-1]) + 1) ||
          (target_intervals[1] - mean(target_intervals[-1])) >
            (sd(target_intervals[-1]) + 1)) {
      drop_first <- TRUE
    } else {
      drop_first <- FALSE
    }
  }

  input <- dplyr::semi_join(input, target, by = "date")
  target <- dplyr::semi_join(target, input, by = "date")

  if (drop_first) {
    input <- input[-1, ]
    target <- target[-1, ]
  }

  if (nrow(input) != nrow(target)) {
    stop("There are state or date mismatches between the input and target data")
  }

  input$daily_std <- (input$daily - mean(input$daily)) /
    sd(input$daily)
  input$elapsed_std <- (input$elapsed - mean(input$elapsed)) /
    sd(input$elapsed)
  input$target_daily_std <- (target$daily - mean(target$daily)) /
    sd(target$daily)

  if (length(unique(input$state)) == 1) {
    model <- brms::brm(target_daily_std ~ daily_std * elapsed_std,
                       data = input,
                       family = gaussian())
  } else {
    model <- brms::brm(target_daily_std ~ daily_std * elapsed_std +
                         (1 + daily_std | state),
                       data = input,
                       family = gaussian())
  }

  print(summary(model))
  plot(model)

  return(model)
}