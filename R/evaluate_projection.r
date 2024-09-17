#' Evaluate a projection model
#'
#' To evaluate how successfully a projection model predicts future uptake,
#' plots of predicted incident and cumulative uptake are generated, along with
#' the data against which the model is being compared. The model ought to
#' match the data closely, although uncertainty may propagate through time.
#' Each plot shows the data and model predictions with credible intervals
#' (95% by default).
#'
#' If only old data, i.e. the data on which the model was fit, is provided,
#' the projections will be compared against that. If new data, i.e. data
#' which the model has not seen, is also provided, projections will compared
#' against that instead. In the latter case, the fit may not be as good, and
#' credible intervals will be wider due to extra uncertainty over the values
#' of year-specific parameters.
#'
#' If no start date is provided, it is assumed the exact incident and
#' cumulative uptake is known for each time t-1 when incident uptake for
#' time t is being predicted. This evaluates how well the model fits
#' each individual data point through the entire time series.
#'
#' If a start date is provided, it is assumed that the exact incident and
#' cumulative uptake is known only until that date. After that date,
#' uncertainty propagates through time, as if actually projecting the future.
#'
#' If an end date is not provided, projections will be provided through
#' the end date of the provided data set.
#'
#' If an end date is provided, projections will be provided through the
#' closest expected reporting date to the end date, even if this extrapolates
#' beyond the end of the provided data set.
#'
#' UPDATES FOR THE FUTURE:
#' - Model must incorporate year-to-year variability
#' - Model must incorporate state-to-state variability
#' - Multiple models can be provided, with scaling, to pool projections
#'
#' @description Evaluate a model that projects uptake into the future
#'
#' @param old_data uptake data used to fit the projection model
#' @param model model fit to project uptake into the future
#' @param new_data uptake data to compare the projection model against
#' @param start_date which date to perform projection from, in %Y-%m-%d format
#' @param end_date which date to perform projections until, in %Y-%m-%d format
#' @param conf width of credible interval to plot
#'
#' @return data frame of the data underlying diagnostic plots
#'
#' @export
#'
evaluate_projection <- function(old_data, model, new_data = NULL,
                                start_date = NULL, end_date = NULL,
                                conf = 0.95) {

  model_rows <- (nrow(old_data) - nrow(model$data) + 1):(nrow(old_data))

  daily_mean <- mean(old_data$daily[model_rows])
  daily_sd <- sd(old_data$daily[model_rows])
  previous_mean <- mean(old_data$daily[model_rows - 1])
  previous_sd <- sd(old_data$daily[model_rows - 1])
  elapsed_mean <- mean(old_data$elapsed[model_rows])
  elapsed_sd <- sd(old_data$elapsed[model_rows])

  if (is.null(new_data)) {
    output <- old_data
  } else {
    output <- new_data
  }

  output$previous <- c(NA, output$daily[1:(nrow(output) - 1)])
  output$interval <- c(output$elapsed[1], diff(output$elapsed))

  if (!is.null(end_date)) {
    end_date <- as.Date(end_date)
    if (end_date <= output$date[nrow(output)]) {
      output <- output[output$date <= end_date, ]
    } else {
      extra_dates <- seq(output$date[nrow(output)], end_date,
                         by = round(mean(output$interval)))[-1]
      extra_rows <- nrow(output) + seq_along(length(extra_dates))
      output[extra_rows, ] <- NA
      output$date[extra_rows] <- extra_dates
      output$elapsed <- as.numeric(output$date -
                                     (output$date[1] - output$elapsed[1]))
      output$previous[extra_rows[1]] <- output$daily[extra_rows[1] - 1]
    }
  }

  output$incident_mean <- rep(NA, nrow(output))
  output$incident_upper <- rep(NA, nrow(output))
  output$incident_lower <- rep(NA, nrow(output))
  output$cumulative_mean <- rep(NA, nrow(output))
  output$cumulative_upper <- rep(NA, nrow(output))
  output$cumulative_lower <- rep(NA, nrow(output))

  if (!is.null(start_date)) {
    start_date <- as.Date(start_date)
    early_dates <- which(output$date <= start_date | is.na(output$previous))
  } else {
    early_dates <- which(is.na(output$previous))
  }
  output$incident_mean[early_dates] <- output$incident[early_dates]
  output$incident_upper[early_dates] <- output$incident[early_dates]
  output$incident_lower[early_dates] <- output$incident[early_dates]
  output$cumulative_mean[early_dates] <- output$cumulative[early_dates]
  output$cumulative_upper[early_dates] <- output$cumulative[early_dates]
  output$cumulative_lower[early_dates] <- output$cumulative[early_dates]

  output_copy <- output[is.na(output$incident_mean), ]

  if (is.null(start_date)) {
    input <- data.frame(elapsed_std = (output_copy$elapsed - elapsed_mean) /
                          elapsed_sd,
                        previous_std = (output_copy$previous - previous_mean) /
                          previous_sd)

    post_check <- brms::posterior_predict(model, newdata = input) *
      daily_sd + daily_mean
  } else {
    output_copy <- output[is.na(output$incident_mean), ]

    post_check <- matrix(NA, nrow = brms::ndraws(model),
                         ncol = nrow(output_copy))

    for (i in seq_len(nrow(output_copy))) {
      if (i == 1) {
        input <- data.frame(elapsed_std = (output_copy$elapsed[i] -
                                             elapsed_mean) / elapsed_sd,
                            previous_std = (output_copy$previous[i] -
                                              previous_mean) / previous_sd)
        post_check[, i] <- as.vector(brms::posterior_predict(model,
                                                             newdata = input))
      } else {
        input <- data.frame(elapsed_std = (output_copy$elapsed[i] -
                                             elapsed_mean) / elapsed_sd,
                            previous_std = (post_check[, i - 1] -
                                              previous_mean) / previous_sd)
        post_check[, i] <- as.vector(brms::posterior_predict(model,
                                                             newdata = input,
                                                             ndraws = 1))
      }
      post_check[, i] <- post_check[, i] * daily_sd + daily_mean
    }
  }

  post_check <- sweep(post_check, 2, output_copy$interval, FUN = `*`)

  output_copy$incident_mean <- colMeans(post_check)
  output_copy$incident_upper <- apply(post_check, 2, quantile,
                                      1 - (1 - conf) / 2)
  output_copy$incident_lower <- apply(post_check, 2, quantile,
                                      (1 - conf) / 2)

  if (is.null(start_date)) {
    post_check <- sweep(post_check, 2,
                        (output_copy$cumulative - output_copy$incident),
                        FUN = `+`)
  } else {
    post_check <- t(apply(post_check, 1, cumsum))
    post_check <- sweep(post_check, 2, (output_copy$cumulative[1] -
                                          output_copy$incident[1]),
                        FUN = `+`)
  }

  output_copy$cumulative_mean <- colMeans(post_check)
  output_copy$cumulative_upper <- apply(post_check, 2, quantile,
                                        1 - (1 - conf) / 2)
  output_copy$cumulative_lower <- apply(post_check, 2, quantile,
                                        (1 - conf) / 2)

  output <- rbind(output[!is.na(output$incident_mean), ], output_copy)

  output$interval <- NULL
  output$previous <- NULL

  incident_plot <- ggplot2::ggplot(output) +
    ggplot2::geom_ribbon(ggplot2::aes(x = date, ymax = incident_upper,
                                      ymin = incident_lower),
                         fill = "black", alpha = 0.25) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = incident),
                       color = "dodgerblue", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = incident_mean),
                       color = "black", linewidth = 1.5) +
    ggplot2::theme_bw() +
    ggplot2::xlab("Time") + ggplot2::ylab("% of Population") +
    ggplot2::theme(text = ggplot2::element_text(size = 20)) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.9 * diff(range(output$incident_upper)),
                      label = "Input", col = "dodgerblue", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.75 * diff(range(output$incident_upper)),
                      label = "Model", col = "black", size = 10)
  print(incident_plot)

  cumulative_plot <- ggplot2::ggplot(output) +
    ggplot2::geom_ribbon(ggplot2::aes(x = date, ymax = cumulative_upper,
                                      ymin = cumulative_lower),
                         fill = "black", alpha = 0.25) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = cumulative),
                       color = "dodgerblue", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = cumulative_mean),
                       color = "black", linewidth = 1.5) +
    ggplot2::theme_bw() +
    ggplot2::xlab("Time") + ggplot2::ylab("% of Population") +
    ggplot2::theme(text = ggplot2::element_text(size = 20)) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.25 * diff(range(output$cumulative_upper)),
                      label = "Input", col = "dodgerblue", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.1 * diff(range(output$cumulative_upper)),
                      label = "Model", col = "black", size = 10)
  print(cumulative_plot)

  return(output)
}