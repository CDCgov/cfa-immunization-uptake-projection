rm(list = ls())
library(dplyr)
library(ggplot2)
library(lme4)

nis = read.csv("data/nis_flu_states.csv")

nis$time_end = as.Date(nis$time_end)

nis %>%
  mutate(time_end=as.Date(time_end),
         season_start_year= gsub("(\\d{4})/\\d{4}","\\1",season)) %>%
  mutate(season_start=as.Date(paste0(season_start_year,"-07-01"))) %>%
  mutate(elapsed=as.numeric(time_end-season_start))%>%
  mutate(season=factor(season))%>%
  group_by(season) %>%
  mutate(inc=c(0,diff(estimate))) %>%
  filter(inc !=0) %>%
  select(-season_start_year) -> nis

nis %>%
  filter(format(time_end, "%m-%d") == "05-31") %>%
  rename(end_of_season=estimate)-> end_of_season

nis %>%
  filter(format(time_end, "%m-%d") != "05-31") %>%
  left_join(end_of_season %>% select(geography, season, end_of_season)) %>%
  filter(!is.na(end_of_season)) -> nis


fit_pop <- lmer(end_of_season ~ elapsed + (1|season) + (1|geography), data = nis)
summary(fit_pop)

fit_pop_inter <- lmer(end_of_season ~ elapsed + (1|season) + (1|geography) + (1|season:geography),
                    data = nis)
summary(fit_pop_inter)

nis %>%
  bind_cols(data.frame(fitted = fitted(fit_pop)), fitted_inter=fitted(fit_pop_inter)) %>%
  ggplot() +
  geom_point(aes(x = elapsed, y = end_of_season, color = season)) +
  geom_line(aes(x = elapsed, y = fitted_inter,color=season)) +
  facet_wrap(geography ~.) +
  theme_bw()

nis %>%
  filter(time_end > as.Date("2023-08-01")) -> nis_fc

nis %>%
  filter(time_end <= as.Date("2023-08-01")) -> nis_train

mspe <- function(test,fc) {

  mean((test-fc)^2)
}

retro_fc <- function(fc_split) {
  nis %>%
    filter(time_end > fc_split) -> nis_test

  nis %>%
    filter(time_end <= fc_split) -> nis_train

  fit_pop <- lmer(end_of_season ~ elapsed + (1|season) + (1|geography), data = nis_train)

  fit_pop_inter <- lmer(end_of_season ~ elapsed + (1|season) + (1|geography) + (1|season*geography),
                      data = nis_train)

  aic_diff = AIC(fit_pop_inter)-AIC(fit_pop)

  fit_pop_fc <- predict(fit_pop,newdata = nis_test, re.form=NA)
  fit_pop_inter_fc <- predict(fit_pop_inter,newdata=nis_test,re.form=NA)

  mspe_pop <- mspe(nis_test$estimate,fit_pop_fc)
  mspe_pop_inter <- mspe(nis_test$estimate, fit_pop_inter_fc)

  fit_pop_train <- predict(fit_pop, nis_train)
  fit_pop_inter_train <- predict(fit_pop_inter, nis_train)

  mspe_pop <- mspe(nis_train$end_of_season, fit_pop_train)
  mspe_pop_inter <- mspe(nis_train$end_of_season, fit_pop_inter_train)

  mspe_diff <- mspe_pop_inter - mspe_pop

  return(data.frame(aic = aic_diff, mspe=mspe_diff))
  # return(aic_diff)
}

fc_dates = unique(nis_fc$time_end)[-length(unique(nis_fc$time_end))]

sel_fc_dates = fc_dates[1:3]
results = lapply(sel_fc_dates,retro_fc)
names(results) = sel_fc_dates

results_df = plyr::ldply(results)
results_df

results_df %>%
  reshape2::melt(".id") %>%
  mutate(fc_date = as.Date(.id),
         variable=factor(variable,levels=c("aic","mspe"),
                         labels=c("AIC difference","MSPE difference"))) %>%
  ggplot() +
  geom_line(aes(x=fc_date,y=value)) +
  scale_x_date(date_breaks = "1 month",name = "forecast date") +
  ylab("") +
  facet_grid(variable ~.,scales="free") +
  theme_bw()
