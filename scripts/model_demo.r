# Scratchwork to test and evaluate models

# Relevant data sets from the NIS API are:
#    ksfb-ug5d: recent covid
#    udsf-9v7b: historical covid
#    sw5n-wg2p: recent flu
#    vh55-3he6: historical flu

# Model structures to consider will be:
#    logistic + linear model

# Load the necessary packages
library(tidyverse)
library(nanoparquet)
library(lubridate)
library(stringr)

# Set the user-controlled parameters
disease <- "flu" # "covid" or "flu"
forecast_date <- as.Date("2022-08-01") # Model was trained only until this date!
end_date <- as.Date("2023-06-30")

# Load recent & historical data, filter to national scale, & declare rollouts
if (disease == "covid") {
    rec <- read_parquet(".cache/nisapi/clean/id=ksfb-ug5d/part-0.parquet")
    rec <- filter(
        rec,
        geography_type == "admin1",
        domain_type == "age",
        domain == "18+ years",
        indicator_type == "4-level vaccination and intent",
        indicator == "received a vaccination",
        time_type == "week"
    )
    rec <- select(rec, time_end, geography, estimate)
    his <- read_parquet(".cache/nisapi/clean/id=udsf-9v7b/part-0.parquet")
    his <- filter(
        his,
        geography_type == "admin1",
        domain_type == "overall",
        indicator_type == "Received updated bivalent booster (among adults who completed primary series)",
        time_type == "week"
    )
    his <- select(his, time_end, geography, estimate)
    first_month <- 9
} else if (disease == "flu") {
    rec <- read_parquet(".cache/nisapi/clean/id=sw5n-wg2p/part-0.parquet")
    rec <- filter(
        rec,
        geography_type == "admin1",
        domain_type == "age",
        domain == "18+ years",
        indicator_type == "4-level vaccination and intent",
        indicator == "received a vaccination",
        time_type == "week"
    )
    rec <- select(rec, time_end, geography, estimate)
    his <- read_parquet(".cache/nisapi/clean/id=vh55-3he6/part-0.parquet")
    his <- filter(
        his,
        vaccine == "flu",
        geography_type == "admin1",
        domain_type == "age",
        domain == "18+ years",
        indicator_type == "uptake",
        indicator == "received a vaccination",
        time_type == "month"
    )
    his <- select(his, time_end, geography, estimate)
    his <- his[his$time_end < as.Date("2023-09-30"), ]
    first_month <- 7
} else {
    print("Disease must be 'covid' or 'flu'")
}

# Combine recent and historical data, sort, and trim
data <- rbind(rec, his)
data <- arrange(distinct(data), time_end)
colnames(data) <- c("date", "geography", "cumulative")
data <- data[data$date < forecast_date, ]

# Add cols: season and elapsed (as fraction of season)
data$season <- as.character(year(ymd(data$date)) -
    ifelse(month(ymd(data$date)) < first_month, 1, 0))
data$elapsed <- as.numeric(data$date - as.Date(paste(data$season,
    "-0", as.character(first_month), "-01",
    sep = ""
))) / 365
data$season <- paste(data$season, as.character(as.numeric(data$season) + 1),
    sep = "/"
)
data$geography <- stringr::str_replace_all(data$geography, " ", "_")

# Load model parameter estimates
if (disease == "flu") {
    post <- read_parquet("data/posteriors_flu_HillModel.parquet")
}
post <- select(post, -model, -forecast_start, -forecast_end, -d, -chain, -draw)

# Generate posterior predictions for each data point
pred <- matrix(0, nrow = nrow(data), ncol = nrow(post))
for (i in 1:nrow(data)) {
    season <- data$season[i]
    state <- data$geography[i]
    elapsed <- data$elapsed[i]
    sub_post <- select(
        post, A, H, n, M,
        A_sigs_season, A_sigs_geography, M_sigs_season, M_sigs_geography,
        A_devs_season = paste("A_devs_", season, sep = ""), A_devs_geography = paste("A_devs_", state, sep = ""),
        M_devs_season = paste("M_devs_", season, sep = ""), M_devs_geography = paste("M_devs_", state, sep = "")
    )
    sub_post <- mutate(sub_post,
        A = A + A_sigs_season * A_devs_season + A_sigs_geography * A_devs_geography,
        M = M + M_sigs_season * M_devs_season + M_sigs_geography * M_devs_geography
    )
    pred[i, ] <- sub_post$A / (1 + exp(-sub_post$n * (elapsed - sub_post$H))) + sub_post$M * elapsed
}

# Make plotting data frame
plot <- data
plot$pred <- rowMeans(pred)
plot$lci <- apply(pred, 1, quantile, 0.025)
plot$uci <- apply(pred, 1, quantile, 0.975)

for (i in unique(plot$season)) {
    sub_plot <- plot[plot$season == i, ]
    p <- ggplot(data = sub_plot, aes(x = date)) +
        geom_point(aes(y = cumulative)) +
        geom_line(aes(y = pred)) +
        geom_ribbon(aes(ymin = lci, ymax = uci)) +
        theme_bw() +
        xlab("Date") +
        ylab("Cumulative Uptake") +
        facet_wrap("geography")
    ggsave(
        filename = paste("data/flu_postcheck_", substr(i, 1, 4), ".png", sep = ""),
        plot = p, width = 9, height = 9, dpi = 300
    )
}
