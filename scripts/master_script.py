from function_development import get_nis
from function_development import ProjectionSettings

# https://data.cdc.gov/Flu-Vaccinations/Weekly-Cumulative-Influenza-Vaccination-Coverage-A/2v3t-r3np/about_data
flu_2023 = get_nis(
    data_path="https://data.cdc.gov/api/views/2v3t-r3np/rows.csv?accessType=DOWNLOAD\u0026bom=true\u0026format=true",
    region_col="Geographic_Name",
    date_col="Current_Season_Week_Ending",
    estimate_col="ND_Weekly_Estimate",
    filters={
        "Geographic_Name": "National",
        "Demographic_Level": "Overall",
        "Demographic_Name": "18+ years",
        "Indicator_Category_Label": "Received a vaccination",
    },
)

first_try = ProjectionSettings(
    flu_2023,
    start_date="2023-12-25",
    end_date="2024-04-30",
    interval="7d",
    rollout_dates="2023-09-01",
)

first_try = first_try.build_model()
# first_try = first_try.evaluate_model() # This is for Fuhan to write
# BEWARE: this example produces a terrible model because
# there is not even 1 full season to learn from
