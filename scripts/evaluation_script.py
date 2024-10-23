from iup.get_uptake_data import get_uptake_data
from iup.get_projections import get_projections
from iup.plot_projections import plot_projections

# Load 2022 NIS data for USA
nis_usa_2022 = get_uptake_data(
    file_name="https://data.cdc.gov/api/views/akkj-j5ru/rows.csv?accessType=DOWNLOAD",
    state_col="Geography",
    date_col=["Time Period", "Year"],
    cumulative_col="Estimate (%)",
    state_key="data/USA_Name_Key.csv",
    date_format="%m/%d/%Y",
    start_date="9/2/2022",
    filters={
        "Indicator Category": "Received updated bivalent booster dose (among adults who completed primary series)"
    },
)

# Load 2023 NIS data for USA
nis_usa_2023 = get_uptake_data(
    file_name="data/NIS_2023-24.csv",
    state_col="geography",
    date_col="date",
    cumulative_col="estimate",
    state_key="data/USA_Name_Key.csv",
    date_format="%m/%d/%Y",
    start_date="9/12/2023",
    filters={"time_type": "Weekly", "group_name": "Overall"},
)

# 2022 NIS data as training, to sequentially (by 1 day) predict 2023 uptake #
rt_pred_2023 = []

for i in range(len(nis_usa_2023) - 1):
    pred_2023 = get_projections(
        train_data=nis_usa_2022,
        data_to_initiate=nis_usa_2023[i],
        end_date=nis_usa_2023["date"][len(nis_usa_2023) - 1],
    )

    rt_pred_2023.append(pred_2023)

# plot the weekly-initiated projections with observed data #
plot_projections(
    obs_data=nis_usa_2023,
    pred_data_list=rt_pred_2023,
    n_columns=6,
    pic_loc="output/",
    pic_name="weekly_predicted_uptake.png",
)
