rm(list=ls())
library(dplyr)
library(ggplot2)
library(mgcv)

### national flu data ###

##### data cleaning #####
nis <- read.csv("../data/nis_flu_national.csv")
head(nis)
nis$time_end = as.Date(nis$time_end)

nis %>%
  mutate(time_end=as.Date(time_end),
         season_start_year= gsub("(\\d{4})/\\d{4}","\\1",season)) %>%
  mutate(season_start=as.Date(paste0(season_start_year,"-07-01"))) %>%
  mutate(elapsed=as.numeric(time_end-season_start))%>%
  mutate(season=factor(season))%>%
  select(-season_start_year) -> nis

nis <- nis[order(nis$time_end),]

train_end <- as.Date("2020-07-01")

#### First 11 seasons as training, the rest 4 seasons as testing ###
nis %>%
  filter(time_end < train_end) -> nis_train

nis %>%
  filter(time_end >= train_end) -> nis_test

##### model fitting ######
# add log-normal link function by manually log-transforming response variable
nis_gam <- gam(log(estimate) ~ s(elapsed,bs="bs") + s(season,bs="re"),data= nis_train)

# transform back #
nis_train$fitted_values = exp(nis_gam$fitted.values)

## Fitted curves shares the same trend
nis_train %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=estimate)) +
  geom_line(aes(x=elapsed,y=fitted_values,color=season)) +
  theme_bw()
ggsave("plot/fitted_value_by_season.jpg",units="in",
       width=6,height=4)


# further looking into fitted terms:
# s(elapsed) is Xbeta, s(season) is group-specific intercept
fitted_terms <- predict(nis_gam, newdata=nis_train,type="terms")

# if plotting fitted Xb only:
fitted_xb <- exp(fitted_terms[,"s(elapsed)"] + coef(nis_gam)[1])

nis_train %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=estimate))+
  geom_line(aes(x=elapsed,y=fitted_xb,color=season)) +
  theme_bw()
ggsave("plot/main_effect_by_season.jpg",units="in",
       width=6,height=4)


### Prediction ###
pred_values <- predict(nis_gam, newdata=nis_test,types="response")
nis_test$pred_terms <- exp(pred_values)

## The group-specific intercept is zero as the new seasons
# are not in the training. In this case, only the mean of
# main effect is predicted
ggplot(nis_test) +
  geom_point(aes(x=elapsed,y=estimate))+
  geom_line(aes(x=elapsed, y=pred_terms,color=season)) +
  theme_bw()
ggsave("plot/pred_by_season.jpg",units="in",width=6,height=4)


## sequentially add season ##
pred_by_season <- function(data, train_end){

  data %>%
    filter(time_end < train_end) -> data_train

  data %>%
    filter(time_end >= train_end) -> data_test

  data_gam <- gam(log(estimate) ~ s(elapsed,bs="bs") + s(season,bs="re"),data= data_train)

  pred <- exp(predict(data_gam, newdata=data_test,type="response"))

  data_test$pred <- pred

  return(data_test)

}

# use 2, 4, 8, 12, 14 seasons as training #
train_ends <- c(as.Date("2011-07-01"),
                as.Date("2013-07-01"), as.Date("2017-07-01"),
                as.Date("2021-07-01"), as.Date("2023-07-01"))

train_ids <- data.frame(id=paste(c(2,4,8,12,14),"seasons training"),
                         train_end=c(as.Date("2011-07-01"),
                                     as.Date("2013-07-01"), as.Date("2017-07-01"),
                                     as.Date("2021-07-01"), as.Date("2023-07-01")))

pred_list <- lapply(1:nrow(train_ids),
                    function(x){pred_by_season(nis, train_end=train_ids$train_end[x])})

names(pred_list) <- train_ids$id

pred_df <- plyr::ldply(pred_list)

pred_df %>%
  mutate(id=factor(.id,levels=paste(c(2,4,8,12,14),"seasons training"))) %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=estimate,color=season)) +
  geom_line(aes(x=elapsed,y=pred)) +
  facet_wrap(id~.) +
  theme_bw()
ggsave("plot/pred_by_sel_seasons.jpg",units="in",width=6,height=4)

# evaluate using mspe
mspe <- function(data, pred) {
  mean((data-pred)^2)
}

pred_df %>%
  group_by(.id) %>%
  reframe(mspe=mspe(estimate,pred)) -> mspe_df

mspe_df %>%
  mutate(id=factor(.id,levels=paste(c(2,4,8,12,14),"seasons training"),
                   labels=paste(c(2,4,8,12,14),"seasons\ntraining"),)) %>%
  ggplot() +
  geom_col(aes(x=id,y=mspe)) +
  theme_bw()
ggsave("plot/mspe_by_sel_seasons.jpg",units="in",width=6,height=4)
