# This is a demo of our candidate model structures

# Relevant data sets from the NIS API are:
#    ksfb-ug5d: recent covid
#    udsf-9v7b: historical covid
#    sw5n-wg2p: recent flu
#    vh55-3he6: historical flu

# Model structures to consider will be:
#    linear incident model (LIM)
#    cumulative Hill model (CHM)

# Predictions will begin in late Oct 2024, to enable:
#    Quantitative evaluation of models vs. observed Nov 2024
#    Visual evaluation of models vs. unobserved May 2025

# Load the necessary packages
library(tidyverse)
library(nanoparquet)
library(lubridate)
library(brms)

# Set the user-controlled parameters
disease <- "flu" # "covid" or "flu"
forecast_date <- as.Date("2024-10-25")
end_date <- as.Date("2025-05-31")

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
    rollouts <- as.Date(c("2022-09-01", "2023-09-22", "2024-09-01"))
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
    rollouts <- as.Date(c(
        "2009-07-01", "2010-07-01", "2011-07-01", "2012-07-01",
        "2013-07-01", "2014-07-01", "2015-07-01", "2016-07-01",
        "2017-07-01", "2018-07-01", "2019-07-01", "2020-07-01",
        "2021-07-01", "2022-07-01", "2023-07-01", "2024-07-01"
    ))
    first_month <- 7
} else {
    print("Disease must be 'covid' or 'flu'")
}

# Combine recent and historical data, insert rollout dates, and sort
data <- rbind(rec, his)
data <- rbind(
    data,
    data.frame(time_end = rollouts, estimate = rep(0, length(rollouts)))
)
data <- arrange(distinct(data), time_end)
colnames(data) <- c("date", "geography", "cumulative")

# Add cols: incident uptake, season, days elapsed, interval, daily avg, previous
# DOES NOT ACCOUNT FOR GEOGRAPHY OR AGE GROUPS
data$incident <- c(0, diff(data$cumulative))
data$incident[data$cumulative == 0] <- 0
data$season <- year(ymd(data$date)) -
    ifelse(month(ymd(data$date)) < first_month, 1, 0)
data$elapsed <- as.numeric(data$date - rollouts[match(
    data$season,
    year(ymd(rollouts))
)])
data$season <- paste(as.character(data$season),
    as.character(data$season + 1),
    sep = "/"
)
data$interval <- c(0, as.numeric(diff(data$date)))
data$interval[data$cumulative == 0] <- 0
data$daily <- data$incident / data$interval
data$daily[data$cumulative == 0] <- 0
data$previous <- c(0, data$daily[1:(nrow(data) - 1)])
data$previous[data$cumulative == 0] <- 0

# Split the data into train and test portions
train <- data[data$date < forecast_date, ]
test <- data[data$date >= forecast_date &
    data$season == train$season[nrow(train)], ]

# Visualize all available data (training & test)
ggplot(
    data = data,
    aes(x = elapsed, y = 100 * cumulative, color = season)
) +
    geom_point() +
    geom_line() +
    theme_bw() +
    theme(text = element_text(size = 15)) +
    xlab("Days Since Rollout") +
    ylab("Cumulative Uptake (%)")

# TEMPORARY PLOT OF DATA BY STATE WITHIN A SEASON
ggplot(
    data = data[data$season == "2020/2021", ],
    aes(x = elapsed_frac, y = 100 * cumulative, color = geography)
) +
    # geom_point() +
    geom_line() +
    theme_bw() +
    theme(text = element_text(size = 15), legend.position = "none") +
    xlab("Days Since Rollout") +
    ylab("Cumulative Uptake (%)")

# TEMPORARY PLOT OF ALL DATA VS. A HILL FIT FROM NUMPYRO: A = 0.42, H = 0.32, N = 5.26
data$elapsed_frac <- data$elapsed / 365
d <- dplyr::select(data, -c(incident, interval, daily, previous, elapsed))
m <- data.frame(elapsed_frac = 0:365 / 365)
m$cumulative <- 0.42 * (m$elapsed_frac^5.26) / (0.32^5.26 + m$elapsed_frac^5.26)
ggplot() +
    geom_point(data = m, aes(x = elapsed_frac, y = 100 * cumulative)) +
    geom_line(data = d, aes(x = elapsed_frac, y = 100 * cumulative, color = season)) +
    theme_bw() +
    theme(text = element_text(size = 15)) +
    xlab("Fraction of Season") +
    ylab("Cumulative Uptake (%)")

# TEMPORARY PLOT OF MIXED LOGISTIC AND LINEAR
x <- seq(0, 1, by = 0.01)
y <- 0.37 / (1 + exp(-25.35 * (x - 0.32))) + 0.06 * x
plot(x, y)

# TEMPORARY PLOT OF HYPERTABASTIC CDF.
sech <- function(x) {
    return(2 / (exp(x) + exp(-x)))
}
coth <- function(x) {
    return((exp(x) + exp(-x)) / (exp(x) - exp(-x)))
}
t <- seq(0.01, 3.65, by = 0.01)
A <- 4
H <- 1
y <- 1 - sech((A / H) * (1 - (t^H) * coth(t^H)))
plot(t, y)

# Tailor training data for LIM: drop first 2 dates per season & standardize
train_lim <- train %>%
    group_by(season) %>%
    slice(-1:-2)
train_lim$daily_std <- scale(train_lim$daily)[, 1]
train_lim$elapsed_std <- scale(train_lim$elapsed)[, 1]
train_lim$previous_std <- scale(train_lim$previous)[, 1]

# Build the LIM
priors <- c(
    prior(normal(0, 0.2), class = "Intercept"),
    prior(normal(0, 0.2), class = "b"),
    prior(cauchy(0, 0.5), class = "sigma")
)
lim <- brms::brm(daily_std ~ previous_std * elapsed_std,
    data = train_lim,
    family = gaussian(),
    prior = priors
)

# Set up data frame for LIM projection
proj_dates <- c(forecast_date, test$date)
lim_proj <- data.frame(date = c(
    proj_dates,
    seq(proj_dates[length(proj_dates)] + 7,
        end_date,
        by = 7
    )
))
lim_proj$season <- train$season[nrow(train)]
lim_proj$elapsed <- as.numeric(lim_proj$date -
    data$date[data$season == lim_proj$season[1] & data$cumulative == 0])
lim_proj$interval <- c(as.numeric(lim_proj$date[1] -
    train$date[nrow(train)]), as.numeric(diff(lim_proj$date)))

# Run trajectories from the LIM, one per posterior draw
proj <- matrix(0, nrow(lim_proj), ndraws(lim))
for (i in 1:nrow(proj)) {
    elapsed_std <- rep(
        (lim_proj$elapsed[i] - mean(train_lim$elapsed)) /
            sd(train_lim$elapsed),
        ncol(proj)
    )
    if (i == 1) {
        previous_std <- rep(
            (train_lim$daily[nrow(train_lim)] -
                mean(train_lim$previous)) /
                sd(train_lim$previous),
            ncol(proj)
        )
    } else {
        previous_std <- (proj[i - 1, ] - mean(train_lim$previous)) /
            sd(train_lim$previous)
    }
    input <- data.frame(elapsed_std = elapsed_std, previous_std = previous_std)
    proj[i, ] <- diag(brms::posterior_predict(lim, newdata = input)) *
        sd(train_lim$daily) + mean(train_lim$daily)
}
proj <- sweep(proj, 1, lim_proj$interval, FUN = `*`)
proj <- apply(proj, 2, cumsum) + train_lim$cumulative[nrow(train_lim)]

# Record mean & conf int for cumulative uptake predicted by LIM
lim_proj$cumulative <- rowMeans(proj)
lim_proj$cumulative_uci <- apply(proj, 1, quantile, 0.975)
lim_proj$cumulative_lci <- apply(proj, 1, quantile, 0.025)

# Build the CHM
hill <- bf(cumulative ~ (A * elapsed^n) / (H^n + elapsed^n),
    A + H + n ~ 1,
    nl = T
)
priors <- c(
    prior(normal(0.3, 0.3), nlpar = "A"),
    prior(normal(50, 20), nlpar = "H"),
    prior(normal(2, 1), nlpar = "n"),
    prior(cauchy(0, 1), class = "sigma")
)
chm <- brms::brm(hill, data = train, prior = priors)

# Set up data frame for CHM projection
chm_proj <- lim_proj[, c("date", "season", "elapsed")]

# Draw 1000 samples per timepoint from the CHM
proj <- t(brms::posterior_predict(chm, newdata = chm_proj, ndraws = 1000))

# Record mean & conf int for cumulative uptake predicted by CHM
chm_proj$cumulative <- rowMeans(proj)
chm_proj$cumulative_uci <- apply(proj, 1, quantile, 0.975)
chm_proj$cumulative_lci <- apply(proj, 1, quantile, 0.025)

# Build the CHM with hyperparameters for season
hill_season <- bf(cumulative ~ (A * elapsed^n) / (H^n + elapsed^n),
    A + H ~ 1 + (1 | season), n ~ 1,
    nl = T
)
chms <- brms::brm(hill_season, data = train, prior = priors)

# Set up data frame for CHM projection
chms_proj <- lim_proj[, c("date", "season", "elapsed")]

# Draw 1000 samples per timepoint from the CHM
proj <- t(brms::posterior_predict(chms, newdata = chms_proj, ndraws = 1000))

# Record mean & conf int for cumulative uptake predicted by CHM
chms_proj$cumulative <- rowMeans(proj)
chms_proj$cumulative_uci <- apply(proj, 1, quantile, 0.975)
chms_proj$cumulative_lci <- apply(proj, 1, quantile, 0.025)

# Combine LIM and CHM predictions for a single plot
lim_proj <- select(lim_proj, -interval, )
lim_proj$model <- rep("LIM", nrow(lim_proj))
chm_proj$model <- rep("CHM", nrow(chm_proj))
chms_proj$model <- rep("CHM by Season", nrow(chms_proj))
proj <- rbind(lim_proj, chm_proj, chms_proj)
data <- select(data, date, season, elapsed, cumulative)
data$cumulative_uci <- data$cumulative
data$cumulative_lci <- data$cumulative
data$model <- "Data"
plot <- rbind(proj, data)
plot <- plot[plot$season == "2024/2025", ]

# Plot projections from both models along with in-season data
ggplot() +
    geom_ribbon(
        data = plot,
        aes(
            x = elapsed, ymin = 100 * cumulative_lci,
            ymax = 100 * cumulative_uci, fill = model
        ), alpha = 0.3,
    ) +
    geom_point(
        data = plot,
        aes(
            x = elapsed, y = 100 * cumulative,
            color = model
        )
    ) +
    geom_line(
        data = plot,
        aes(
            x = elapsed, y = 100 * cumulative,
            color = model
        )
    ) +
    ggtitle(paste("Data and projections for ", disease, sep = "")) +
    theme_bw() +
    theme(text = element_text(size = 15)) +
    scale_color_manual(
        values = c("lightpink", "firebrick", "black", "dodgerblue"),
        name = ""
    ) +
    scale_fill_manual(
        values = c("lightpink", "firebrick", "black", "dodgerblue"),
        name = ""
    ) +
    geom_vline(
        xintercept = min(as.numeric(forecast_date -
            rollouts)[as.numeric(forecast_date - rollouts) > 0]),
        linetype = "dashed"
    ) +
    xlab("Days Since Rollout") +
    ylab("Cumulative Uptake (%)")

# Evaluate LIM performance on final observed date, 2024-11-30
lim_eval <- inner_join(test[, c("date", "cumulative")],
    lim_proj[, c("date", "cumulative")],
    by = "date"
)
mspe <- mean((100 * lim_eval$cumulative.x -
    100 * lim_eval$cumulative.y)^2)
print(paste("LIM MSPE: ", round(mspe, 2), sep = ""))
eos_abe <- 100 * (lim_eval$cumulative.y -
    lim_eval$cumulative.x)[nrow(lim_eval)]
print(paste("LIM EOS_ABE: ", round(eos_abe, 2), "%", sep = ""))

# Evaluate CHM performance on final observed date, 2024-11-30
chm_eval <- inner_join(test[, c("date", "cumulative")],
    chm_proj[, c("date", "cumulative")],
    by = "date"
)
mspe <- mean((100 * chm_eval$cumulative.y -
    100 * chm_eval$cumulative.x)^2)
print(paste("CHM MSPE: ", round(mspe, 2), sep = ""))
eos_abe <- 100 * (chm_eval$cumulative.y -
    chm_eval$cumulative.x)[nrow(chm_eval)]
print(paste("CHM EOS_ABE: ", round(eos_abe, 2), "%", sep = ""))
