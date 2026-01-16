rm(list = ls())
library(dplyr)
library(ggplot2)
library(brms)

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


# prs <- c(set_prior("beta(100,180)", nlpar="A", class="Intercept"))

tick <- Sys.time()
fit <- brm(bf(end_of_season ~ elapsed + (1|season) + (1|geography)),
                  family=lognormal(),
                  data=nis,
                  chains=4,cores=4)

fit_interaction <- brm(bf(end_of_season ~ elapsed + (1|season) + (1|geography) + (1|season:geography)),
                              family=lognormal(),
                              data=nis,
                              chains=4,cores=4)

tock <- Sys.time()

tock - tick

save(fit,file="brms_linear_fit.Rdata")
save(fit_interaction,file="brms_linear_fit_interaction.Rdata")


### visual check ###
nis %>%
  bind_cols(data.frame(fit=predict(fit),fit_inter=predict(fit_interaction))) -> all

all %>%
  ggplot() +
  geom_point(aes(x = elapsed, y = end_of_season, color = season), size = 1) +
  geom_line(aes(x = elapsed, y = fit.Estimate)) +
  facet_wrap(geography ~.) +
  theme_bw()

### Diagnostics ###
plot(fit) # model converged
plot(fit_interaction) # model didn't converge

pp_check(fit)
