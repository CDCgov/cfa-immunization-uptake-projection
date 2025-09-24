rm(list=ls())
library(dplyr)
library(ggplot2)
library(mgcv)

### national flu data ###

##### data cleaning #####
nis <- read.csv("data/nis_flu_national.csv")
head(nis)
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

nis <- nis[order(nis$time_end),]

train_end <- as.Date("2023-07-01")

#### First 11 seasons as training, the rest 4 seasons as testing ###
nis %>%
  filter(time_end < train_end) -> nis_train

nis %>%
  filter(time_end >= train_end) -> nis_test

##### model fitting ######
# add log-normal link function by manually log-transforming response variable

gam_k <- function(k) {
  nis_gam <- gam(log(estimate) ~ s(elapsed,bs="bs",k=k) + s(season,bs="re"),data= nis_train)

  return(nis_gam)
}

get_pred_k <- function(k) {
  nis_gam <- gam(log(estimate) ~ s(elapsed,bs="bs",k=k) + s(season,bs="re"),data= nis_train)

  ### Prediction ###
  pred_values <- predict(nis_gam, newdata=nis_test,types="response")
  nis_test$pred_terms <- exp(pred_values)

  return(nis_test)
}

mspe <- function(data, pred) {
  mean((data-pred)^2)
}

### get model ###
gam_list <- lapply(5:15,function(x) {gam_k(x)})

lapply(1:length(gam_list),function(x) {gam_list[[x]]$aic})
lapply(1:length(gam_list),function(x) {gam_list[[x]]$edf})

fitted_k <- lapply(1:length(gam_list),function(x) {
  nis_train$fitted = exp(gam_list[[x]]$fitted.values)
                   return(nis_train)})

names(fitted_k) <- paste0("k=",c(5:15))

fitted_k_df <- plyr::ldply(fitted_k)

fitted_k_df %>%
  mutate(.id=factor(.id,levels=paste0("k=",c(5:15)))) %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=estimate)) +
  geom_line(aes(x=elapsed,y=fitted,color=season)) +
  facet_wrap(.id~.) +
  theme_bw()
ggsave("plot/gam_k_cumu_fitted.jpg",width=10,height=6)

pred_k <- lapply(5:15, function(x) {get_pred_k(x)})
names(pred_k) = paste0("k=",c(5:15))

pred_k_df <- plyr::ldply(pred_k)

pred_k_df %>%
  mutate(.id=factor(.id,levels=paste0("k=",c(5:15)))) %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=estimate)) +
  geom_line(aes(x=elapsed,y=pred_terms)) +
  facet_wrap(.id~.) +
  theme_bw()
ggsave("plot/gam_k_cumu_pred.jpg",width=10,height=6)

pred_k_df %>%
  group_by(.id) %>%
  reframe(mspe=mspe(estimate,pred_terms)) %>%
  mutate(id=factor(.id,levels=paste0("k=",c(5:15)))) -> mspe_cum

mspe_cum %>%
  ggplot() +
  geom_col(aes(x=id,y=mspe)) +
  theme_bw()
ggsave("plot/mspe_cum_gam_k.jpg",width=6,height=4)

############# try on incident data ###########
gam_inc_k <- function(k){
  nis_gam <- gam(log(inc) ~ s(elapsed,bs="bs",k=k) + s(season,bs="re"),data= nis_train)

  return(nis_gam)
}

get_pred_inc_k <- function(k) {

  nis_gam <- gam(log(inc) ~ s(elapsed,bs="bs",k=k) + s(season,bs="re"),data= nis_train)

  ### Prediction ###
  pred_values <- predict(nis_gam, newdata=nis_test,types="response")
  nis_test$pred_terms <- exp(pred_values)

  return(nis_test)
}

### get model ###
gam_list <- lapply(5:15,function(x) {gam_inc_k(x)})

lapply(1:length(gam_list),function(x) {gam_list[[x]]$aic})
lapply(1:length(gam_list),function(x) {gam_list[[x]]$edf})

fitted_k <- lapply(1:length(gam_list),function(x) {
  nis_train$fitted = exp(gam_list[[x]]$fitted.values)
  return(nis_train)})

names(fitted_k) <- paste0("k=",c(5:15))

fitted_k_df <- plyr::ldply(fitted_k)

fitted_k_df %>%
  mutate(.id=factor(.id,levels=paste0("k=",c(5:15)))) %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=inc)) +
  geom_line(aes(x=elapsed,y=fitted,color=season)) +
  facet_wrap(.id~.) +
  theme_bw()
ggsave("plot/gam_k_cumu_fitted.jpg",width=10,height=6)

pred_k <- lapply(5:15, function(x) {get_pred_inc_k(x)})
names(pred_k) = paste0("k=",c(5:15))

pred_k_df <- plyr::ldply(pred_k)

pred_k_df %>%
  mutate(.id=factor(.id,levels=paste0("k=",c(5:15)))) %>%
  ggplot() +
  geom_point(aes(x=elapsed,y=inc)) +
  geom_line(aes(x=elapsed,y=pred_terms)) +
  facet_wrap(.id~.) +
  theme_bw()
ggsave("plot/gam_k_cumu_pred.jpg",width=10,height=6)

pred_k_df %>%
  group_by(.id) %>%
  reframe(mspe=mspe(inc,pred_terms)) %>%
  mutate(id=factor(.id,levels=paste0("k=",c(5:15)))) -> mspe_inc

mspe_inc %>%
  ggplot() +
  geom_col(aes(x=id,y=mspe)) +
  theme_bw()
ggsave("plot/mspe_inc_gam_k.jpg",width=6,height=4)

mspe_inc %>%
  mutate(type="incidental") %>%
  bind_rows(mspe_cum %>%
              mutate(type="cumulative")) %>%
  ggplot() +
  geom_col(aes(x=id,y=mspe)) +
  facet_wrap(type~.) +
  theme_bw()
ggsave("plot/all_mspe_gam_k.jpg",width=8,height=6)
