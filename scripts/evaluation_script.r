rm(list = ls())
# read all the R files under R folder #
R_path <- "R/"
purrr::map(paste0(R_path, list.files("R/")), source)

# load observed data #

# Load 2022 NIS data for USA
nis_usa_2022 <- get_uptake_data(
    "https://data.cdc.gov/api/views/akkj-j5ru/rows.csv?accessType=DOWNLOAD",
    state = "Geography",
    date = c("Time.Period", "Year"),
    cumulative = "Estimate....",
    state_key = "data/USA_Name_Key.csv",
    date_format = "%m/%d/%Y",
    start_date = "9/2/2022",
    filters = list(c(
        "Indicator.Category",
        "Received updated bivalent booster dose (among adults who completed primary series)"
    ))
)

# Load 2023 NIS data for USA
nis_usa_2023 <- get_uptake_data(
    "data/NIS_2023-24.csv",
    state = "geography",
    date = "date",
    cumulative = "estimate",
    state_key = "data/USA_Name_Key.csv",
    date_format = "%m/%d/%Y",
    start_date = "9/12/2023",
    filters = list(c("time_type", "Weekly"), c("group_name", "Overall"))
)


# generate projections initiated at different time #
all_data <- rbind(nis_usa_2022, nis_usa_2023)

# make sure there at least 4 data points for model training: 30 min to run #
plot_real_time_projection(
    data_option = "rate",
    all_data = all_data
)

# evaluate MSPE between projections and data initiated at different time: 30 min to run #
evaluate_real_time_projection(evaluate_option = "mspe", data_option = "rate", all_data = all_data) -> mspe_df

mspe_df %>%
    mutate(forecast_date = forecast_date + as.Date("1970-01-01")) %>%
    filter(mspe < 1) %>%
    ggplot() +
    geom_point(aes(x = forecast_date, y = mspe)) +
    xlab("Forecast date") +
    ylab("MSPE") +
    theme_bw()
ggsave("mspe.jpeg", width = 4, height = 3, units = "in")
# Note for future upgrade: seperate model fitting from plotting and evaluation.
# Repeatedly fitting the model to plot and evaluate is not time-efficient.
