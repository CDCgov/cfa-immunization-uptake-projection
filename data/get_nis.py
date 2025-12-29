"""
Download data from https://data.cdc.gov/Flu-Vaccinations/Influenza-Vaccination-Coverage-for-All-Ages-6-Mont/vh55-3he6/about_data
"""

import nisapi
import polars as pl

data = (
    nisapi.get_nis()
    .filter(
        pl.col("vaccine") == pl.lit("flu"),
        pl.col("geography_type").is_in(["nation", "admin1"]),
        pl.col("domain_type") == pl.lit("age & possible risk"),
        pl.col("domain") == pl.lit(">=18 years"),
        pl.col("time_type") == pl.lit("month"),
        pl.col("indicator_type") == pl.lit("received a vaccination"),
        pl.col("indicator") == pl.lit("yes"),
        pl.col("id") == pl.lit("vh55-3he6"),
    )
    .select(
        [
            "geography_type",
            "geography",
            "time_end",
            "estimate",
            "lci",
            "uci",
            "sample_size",
        ]
    )
    .sort(["geography_type", "geography", "time_end"])
    .collect()
)

data.write_parquet("data/raw.parquet")
