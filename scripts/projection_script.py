from get_uptake_data import get_uptake_data
from build_projection_model import build_projection_model
from make_projections import make_projections

# Load 2022 IIS data for USA
iis_usa_2022 = get_uptake_data(
    file_name="https://data.cdc.gov/api/views/unsk-b7fc/rows.csv?accessType=DOWNLOAD",
    state_col="Location",
    date_col="Date",
    cumulative_col="Bivalent_Booster_18Plus_Pop_Pct",
    state_key="USA_Name_Key.csv",
    date_format="%m/%d/%Y",
    start_date="9/2/2022",
)

# Load 2022 NIS data for USA
nis_usa_2022 = get_uptake_data(
    file_name="https://data.cdc.gov/api/views/akkj-j5ru/rows.csv?accessType=DOWNLOAD",
    state_col="Geography",
    date_col=["Time Period", "Year"],
    cumulative_col="Estimate (%)",
    state_key="USA_Name_Key.csv",
    date_format="%m/%d/%Y",
    start_date="9/2/2022",
    filters={
        "Indicator Category": "Received updated bivalent booster dose (among adults who completed primary series)"
    },
)

# Load 2023 NIS data for USA
nis_usa_2023 = get_uptake_data(
    file_name="NIS_2023-24.csv",
    state_col="geography",
    date_col="date",
    cumulative_col="estimate",
    state_key="USA_Name_Key.csv",
    date_format="%m/%d/%Y",
    start_date="9/12/2023",
    filters={"time_type": "Weekly", "group_name": "Overall"},
)

# Build and use a forward projection model for the IIS 2022 data
iis_usa_2022_model = build_projection_model(iis_usa_2022)
iis_usa_2022_proj = make_projections(
    iis_usa_2022, iis_usa_2022_model, "2022-10-19", "2023-05-10", "1w", 47, 7, 1.7, 7.3
)

# Build and use a forward projection model for the NIS 2022 data
nis_usa_2022_model = build_projection_model(nis_usa_2022)
nis_usa_2022_proj = make_projections(
    nis_usa_2022, nis_usa_2022_model, "2022-09-17", "2023-06-30", "1w", 15, 7, 1.6, 2.2
)

# Build and use a forward projection model for the NIS 2023 data
nis_usa_2023_model = build_projection_model(nis_usa_2023)
nis_usa_2023_proj = make_projections(
    nis_usa_2023,
    nis_usa_2023_model,
    "2022-10-07",
    "2023-05-11",
    "1w",
    25,
    7,
    1.3101,
    4.299,
)
