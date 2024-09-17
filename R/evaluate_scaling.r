#' Evaluate a scaling model
#'
#' To evaluate how successfully a scaling model scales one uptake data set
#' onto another, plots of the incident and cumulative uptake are generated.
#' Each plot shows the input data, the target data, and the model predictions
#' and credible interval (95% by default).
#'
#'UPDATES FOR THE FUTURE:
#' - Model must incorporate year-to-year variability
#'
#' @description Evaluate a model that scales one uptake data set to another
#'
#' @param input uptake data to use as a predictor
#' @param target uptake data to use as the outcome
#' @param model model fit to scale input to target
#' @param conf width of credible interval to plot
#'
#' @return data frame of the data underlying diagnostic plots
#'
#' @export
#'
evaluate_scaling <- function(input, target, model, conf = 0.95) {

  output <- dplyr::inner_join(input, target,
                              by = c("state", "date", "elapsed", "year"))
  output <- output[order(output$date), ]
  colnames(output) <- gsub("\\.x", "_input", colnames(output))
  colnames(output) <- gsub("\\.y", "_target", colnames(output))

  if (nrow(output) == (nobs(model) + 1)) {
    intervals <- diff(output$elapsed)
    output <- output[-1, ]
  } else {
    intervals <- c(output$elapsed[1], diff(output$elapsed))
  }

  known <- output$cumulative_target - output$incident_target

  post_check <- brms::posterior_predict(model) *
    sd(output$daily_target) + mean(output$daily_target)

  post_check <- sweep(post_check, 2, intervals, FUN = `*`)

  output$incident_pred_mean <- colMeans(post_check)
  output$incident_pred_upper <- apply(post_check, 2, quantile,
                                      1 - (1 - conf) / 2)
  output$incident_pred_lower <- apply(post_check, 2, quantile,
                                      (1 - conf) / 2)

  post_check <- sweep(post_check, 2, known, FUN = `+`)

  output$cumulative_pred_mean <- colMeans(post_check)
  output$cumulative_pred_upper <- apply(post_check, 2, quantile,
                                        1 - (1 - conf) / 2)
  output$cumulative_pred_lower <- apply(post_check, 2, quantile,
                                        (1 - conf) / 2)

  incident_plot <- ggplot2::ggplot(output) +
    ggplot2::geom_ribbon(ggplot2::aes(x = date, ymax = incident_pred_upper,
                                      ymin = incident_pred_lower),
                         fill = "black", alpha = 0.25) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = incident_input),
                       color = "dodgerblue", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = incident_target),
                       color = "darkred", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = incident_pred_mean),
                       color = "black", linewidth = 1.5) +
    ggplot2::theme_bw() +
    ggplot2::xlab("Time") + ggplot2::ylab("% of Population") +
    ggplot2::theme(text = ggplot2::element_text(size = 20)) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.9 * diff(range(output$incident_pred_upper)),
                      label = "Input", col = "dodgerblue", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.75 * diff(range(output$incident_pred_upper)),
                      label = "Target", col = "darkred", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.6 * diff(range(output$incident_pred_upper)),
                      label = "Model", col = "black", size = 10)
  print(incident_plot)

  cumulative_plot <- ggplot2::ggplot(output) +
    ggplot2::geom_ribbon(ggplot2::aes(x = date, ymax = cumulative_pred_upper,
                                      ymin = cumulative_pred_lower),
                         fill = "black", alpha = 0.25) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = cumulative_input),
                       color = "dodgerblue", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = cumulative_target),
                       color = "darkred", linewidth = 1.5) +
    ggplot2::geom_line(ggplot2::aes(x = date, y = cumulative_pred_mean),
                       color = "black", linewidth = 1.5) +
    ggplot2::theme_bw() +
    ggplot2::xlab("Time") + ggplot2::ylab("% of Population") +
    ggplot2::theme(text = ggplot2::element_text(size = 20)) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.4 * diff(range(output$cumulative_pred_upper)),
                      label = "Input", col = "dodgerblue", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.25 * diff(range(output$cumulative_pred_upper)),
                      label = "Target", col = "darkred", size = 10) +
    ggplot2::annotate("text", x = output$date[round(0.9 * nrow(output))],
                      y = 0.1 * diff(range(output$cumulative_pred_upper)),
                      label = "Model", col = "black", size = 10)
  print(cumulative_plot)

  return(output)
}