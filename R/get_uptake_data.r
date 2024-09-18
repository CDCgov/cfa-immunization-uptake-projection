#' Format uptake data
#'
#' Download a local or web .csv file of cumulative uptake percentages.
#' You must name the columns that contain the state, date, and uptake.
#'
#' Provide a .csv with state abbreviations in the first column and full
#' names in the second. States will be renamed to their abbreviations,
#' and states missing from the .csv will be discarded.
#'
#' Also provide the format of the dates in the date column. Use the default
#' if there are multiple columns and/or the dates are specified non-numerically.
#' Provide the first date vaccines were available to calculate elapsed time.
#'
#' Optionally, provide filters to remove certain rows from the data.
#' Filters are a list of vectors. In each vector, the first entry gives the name
#' of the column to filter on. Subsequent entries give acceptable values in that
#' column. Rows with unmentioned values will be removed.
#'
#' Incident percentage uptake is calculated from the cumulative uptake.
#' Therefore, if the time intervals are not equivalent between all cumulative
#' reports, a warning will be generated.
#'
#' The uptake data is returned in a data frame with seven columns:
#'
#' Column 1 is the state for which percentage cumulative uptake is reported.
#' Column 2 is the date at which percentage cumulative uptake is reported.
#' Column 3 is the days elapsed since the the vaccine became available.
#' Column 4 is the year of reporting during the fall portion of the season.
#' Column 5 is the percentage cumulative uptake.
#' Column 6 is the incident uptake since the previous report.
#' Column 7 is the incident uptake averaged daily since the previous report.
#'
#' @description Download uptake data and put it into a common format
#'
#' @param file_name file name or url address where the data is located
#' @param state name of column containing the state, or the value to fill in
#' @param date name of column(s) containing the date, or the value to fill in
#' @param cumulative name of column containing cumulative uptake
#' @param state_key file name of .csv with abbr and full name of desired states
#' @param date_format format of the date in its column
#' @param start_date date when vaccines were first available
#' @param filters list of vectors giving column names and values to filter on
#'
#' @return formatted uptake data
#'
#' @export
#'
get_uptake_data <- function(file_name, state, date, cumulative,
                     state_key = NULL, date_format = "%m/%d/%Y",
                     start_date = NULL, filters = NULL) {
  data <- read.csv(file_name)

  if (!is.null(filters)) {
    for (i in seq_along(filters)) {
      if (filters[[i]][1] %in% colnames(data)) {
        data <- data[data[, filters[[i]][1]] %in% filters[[i]][-1], ]
      } else {
        stop(paste("There is no column called ", filters[[i]][1], sep = ""))
      }
    }
  }

  if (cumulative %in% colnames(data)) {
    data <- data[!is.na(data[, cumulative]), ]
    cuml_col <- data[, cumulative]
  } else {
    stop(paste("There is no column called ", cumulative, sep = ""))
  }

  if (state %in% colnames(data)) {
    state_col <- data[, state]
  } else {
    state_col <- rep_len(state, nrow(data))
  }

  if (!is.null(state_key)) {
    state_key <- read.csv(state_key)
    state_idx <- match(state_col, state_key[, 2], nomatch = 0) +
      match(state_col, state_key[, 1], nomatch = 0)
    state_idx[state_idx == 0] <- NA
    state_col <- state_key[, 1][state_idx]
    state_col <- state_col[!is.na(state_col)]
    data <- data[!is.na(state_idx), ]
    cuml_col <- cuml_col[!is.na(state_idx)]
  }

  if (all(date %in% colnames(data))) {
    date_col <- data[, date]
    if (length(date) > 1) {
      for (i in seq_along(date)) {
        if (any(grepl("^[A-Za-z]", date_col[, i]))) {
          date_col[, i] <- sub(".*-", "", date_col[, i])
          date_col[, i] <- trimws(date_col[, i], "both")
          date_col[, i] <- sub(" ", "/", date_col[, i])
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "January", "1")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "February", "2")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "March", "3")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "April", "4")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "May", "5")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "June", "6")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "July", "7")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "August", "8")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "September", "9")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "October", "10")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "November", "11")
          date_col[, i] <- stringr::str_replace_all(date_col[, i],
                                                    "December", "12")
        }
      }
      date_col <- do.call(paste, c(date_col, sep = "/"))
    }
  } else {
    date_col <- rep_len(date, nrow(data))
  }
  date_col <- as.Date(date_col, format = date_format)

  elapsed_col <- as.numeric(date_col - as.Date(start_date,
                                               format = date_format))

  year_col <- as.numeric(format(date_col, "%Y")) -
    1 * (as.numeric(format(date_col, "%m")) < 7)

  data <- data.frame(state = state_col,
                     date = date_col,
                     elapsed = elapsed_col,
                     year = year_col,
                     cumulative = cuml_col)
  data <- data[order(data$date), ]
  
  data$incident <- ave(data$cumulative, data$state,
                       FUN = function(x) {c(x[1], diff(x))})

  data$daily <- data$incident / c(data$elapsed[1], diff(data$elapsed))

  check_intervals <- ave(data$elapsed, data$state,
                         FUN = function(x) {c(x[1], diff(x))})
  if (length(unique(check_intervals)) > 1) {
    warning("Not all time intervals are the same for incident uptake!")
  }

  return(data)
}
