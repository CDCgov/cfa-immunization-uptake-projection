#' Interpolate to match dates across data sets
#'
#' BEWARE: THIS DOES NOT YET ACCOUNT FOR MULTI-YEAR DATA SETS
#'
#' Dates may be offset between two data sets. To make the dates match,
#' linear interpolation is performed on the input data set to align its
#' dates with the target data set. Only target dates occurring between
#' two input dates are included. Target dates outside the input range are
#' discarded.
#'
#' @description Interpolate one uptake data set to match dates with another
#'
#' @param input uptake data on which to perform interpolation
#' @param target uptake data with the dates to match
#'
#' @return interpolated uptake data
#'
#' @export
#'
interpolate_dates <- function(input, target) {
  output <- target
  output$cumulative <- rep(NA, nrow(output))

  for (i in seq_len(nrow(target))) {
    input_sub <- input[input$state == target$state[i], ]
    time_diff <- target$date[i] - input_sub$date
    time_idx <- order(abs(time_diff))[1:2]
    time_diff <- time_diff[time_idx]
    if (sum(time_diff < 0) == 1) {
      input_sub <- input_sub[time_idx, ]
      weights <- as.numeric(abs(time_diff)) / as.numeric(sum(abs(time_diff)))
      output$cumulative[i] <- sum(input_sub$cumulative * rev(weights))
    }
  }

  output <- output[!is.na(output$cumulative), ]

  output$incident <- ave(output$cumulative, output$state,
                         FUN = function(x) {c(x[1], diff(x))})

  output$daily <- output$incident / c(output$elapsed[1], diff(output$elapsed))

  return(output)
}
