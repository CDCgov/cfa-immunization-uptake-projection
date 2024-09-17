setwd("/home/tec0/covid_booster_uptake")
source("get_uptake_data.r")
source("interpolate_dates.r")
source("build_scaling_model.r")
source("evaluate_scaling.r")
source("build_projection_model.r")
source("evaluate_projection.r")
source("projection_functions.r")

# Load 2022 IIS data for USA
iis_usa_2022 <- get_uptake_data(
  "https://data.cdc.gov/api/views/unsk-b7fc/rows.csv?accessType=DOWNLOAD",
  state = "Location",
  date = "Date",
  cumulative = "Bivalent_Booster_18Plus_Pop_Pct",
  state_key = "USA_Name_Key.csv",
  date_format = "%m/%d/%Y",
  start_date = "9/2/2022"
)

# Load 2022 NIS data for USA
nis_usa_2022 <- get_uptake_data(
  "https://data.cdc.gov/api/views/akkj-j5ru/rows.csv?accessType=DOWNLOAD",
  state = "Geography",
  date = c("Time.Period", "Year"),
  cumulative = "Estimate....",
  state_key = "USA_Name_Key.csv",
  date_format = "%m/%d/%Y",
  start_date = "9/2/2022",
  filters = list(c("Indicator.Category",
                   "Received updated bivalent booster dose (among adults who completed primary series)"))
)

# Load 2023 NIS data for USA
nis_usa_2023 <- get_uptake_data(
  "NIS_2023-24.csv",
  state = "geography",
  date = "date",
  cumulative = "estimate",
  state_key = "USA_Name_Key.csv",
  date_format = "%m/%d/%Y",
  start_date = "9/12/2023",
  filters = list(c("time_type", "Weekly"), c("group_name", "Overall"))
)

# Interpolate 2022 IIS dates to match 2022 NIS
iis_usa_2022_interp <- interpolate_dates(iis_usa_2022, nis_usa_2022)

# Build and evaluate a model to scale 2022 IIS to 2022 NIS
iis_to_nis_model <- build_scaling_model(iis_usa_2022_interp, nis_usa_2022)
evaluate_scaling(iis_usa_2022_interp, nis_usa_2022, iis_to_nis_model)

# Build and evaluate a model to project 2022 IIS data forward
iis_forward_model <- build_projection_model(iis_usa_2022)
evaluate_projection(iis_usa_2022, iis_forward_model)
evaluate_projection(iis_usa_2022, iis_forward_model, start_date = "2022-10-19")

# Build and evaluate a model to project 2022 NIS data forward
nis_forward_model <- build_projection_model(nis_usa_2022)
evaluate_projection(nis_usa_2022, nis_forward_model)
evaluate_projection(nis_usa_2022, nis_forward_model, start_date = "2022-09-10")

# Does the 2022 NIS projection model work on the 2023 NIS data?
evaluate_projection(nis_usa_2022, nis_forward_model, nis_usa_2023)
evaluate_projection(nis_usa_2022, nis_forward_model, nis_usa_2023, start_date = "2022-09-30")
