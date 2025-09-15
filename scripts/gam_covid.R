#### gam for covid vaccine uptake ####
rm(list=ls())
library(dplyr)
library(ggplot2)
library(mgcv)

covid <- read.csv("../data/nis_covid_national.csv")

covid %>%
  mutate(time_end=as.Date(time_end),
         season_start_year= gsub("(\\d{4})/\\d{4}","\\1",season)) %>%
  mutate(season_start=as.Date(paste0(season_start_year,"-07-01"))) %>%
  mutate(elapsed=as.numeric(time_end-season_start))%>%
  mutate(season=factor(season))%>%
  select(-season_start_year) -> covid

train_end <- as.Date("2024-09-01")

covid %>%
  mutate(time_end=as.Date(time_end)) %>%
  filter(time_end < train_end) -> covid_train

covid %>%
  mutate(time_end=as.Date(time_end)) %>%
  filter(time_end >= train_end) -> covid_test

covid_gam <- gam(estimate~s(elapsed,bs="bs")+s(season,bs="re"),
                 method="REML",data=covid_train)

covid_train$fitted <- covid_gam$fitted.values

covid_test$pred <- predict(covid_gam,newdata=covid_test,type="response")

ggplot(covid_train) +
  geom_point(aes(x=elapsed,y=estimate,color=season))+
  geom_line(aes(x=elapsed,y=fitted,color=season)) +
  theme_bw()
ggsave("plot/covid_train.jpg",units="in",
       width=5,height=4)

ggplot(covid_test) +
  geom_point(aes(x=elapsed,y=estimate,color=season))+
  geom_line(aes(x=elapsed,y=pred,color=season)) +
  theme_bw()
ggsave("plot/covid_test.jpg",units="in",
       width=5,height=4)
