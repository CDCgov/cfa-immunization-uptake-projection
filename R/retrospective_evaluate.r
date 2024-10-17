library(dplyr)

# Retrospective evaluation of forecast
# MSPE only in this version.
# Update in the future:
# 1. Public health oriented metrics: end-of-season totals, high demand period, etc.
# 2. Probabilistic metrics: QS, WIS, etc.


#' @description Get mean-squared prediction error
#' @param data: observed data, can be a constant or a vector
#' @param pred: the prediction output from models, can be a constant or a vector
#' @return MSPE
get_mspe <- function(data, pred) {
    mean((data - pred)^2)
}

#' @description Retrospectively evaluate the forecast given the test data
#' @param data_option: which uptake data type to evaluate? 'rate' or 'cumulative'
#' @param evaluate_option: which metrics to use? 'mspe' only so far.
#' Adding this indicates changing the output from the embedded functions.
#' @param all_data: Data used to iteratively train and test,
#' with each iteration moving forward one data point
#' @param min_data_size: the minimum data size required for model training,
#' So far, this number is the same for model testing.
#' @return a data frame with MSPE given different initializing date

evaluate_real_time_projection <- function(
    data_option = "rate",
    evaluate_option = "mspe",
    all_data, min_data_size = 8) {
    start_date <- sort(all_data$date)[1 + min_data_size - 1]
    end_date <- sort(all_data$date)[nrow(all_data) - min_data_size + 1]

    date_series <- all_data %>%
        filter(date >= start_date, date <= end_date) %>%
        select(date)

    # convert single-column data.frame to vector #
    date_series <- as.vector(date_series$date) + as.Date("1970-01-01")
    metrics <- data.frame()

    for (split_date in date_series) {
        train_data <- all_data %>%
            filter(date < split_date)

        test_data <- all_data %>%
            filter(date >= split_date)

        output <- real_time_projection(
            train_data = train_data,
            test_data = test_data
        )

        if (data_option == "rate") {
            data_to_eval <- output$rate
        } else if (data_option == "cumulative") {
            data_to_eval <- output$cumulative
        } else {
            stop("Data_option is not valid.")
        }

        ## evaluation ##

        # MSPE #
        # So far, only use MSPE.
        # Note: What can be the code structure if multiple metrics are evaluated simultaneously
        if (evaluate_option == "mspe") {
            metric <- get_mspe(data_to_eval$obs, data_to_eval$mean)
        }

        metrics <- rbind(
            metrics,
            data.frame(
                forecast_date = split_date,
                mspe = metric
            )
        )
    }

    return(metrics)
}
