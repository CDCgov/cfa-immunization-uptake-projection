import argparse

import nisapi
import polars as pl


def preprocess(raw_data: pl.LazyFrame) -> pl.DataFrame:
    return (
        raw_data.filter(
            pl.col("geography_type").is_in(["nation", "admin1"]),
            pl.col("domain_type") == pl.lit("age"),
            pl.col("domain") == pl.lit("18+ years"),
            pl.col("indicator") == pl.lit("received a vaccination"),
        )
        .drop(["indicator_type", "indicator"])
        .head()
        .collect()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", help="clean cache directory")
    p.add_argument("--output", help="output parquet file")
    args = p.parse_args()

    raw_data = nisapi.get_nis(path=args.cache)
    clean_data = preprocess(raw_data)
    clean_data.write_parquet(args.output)
