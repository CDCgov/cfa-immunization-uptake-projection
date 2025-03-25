import argparse
from typing import List

import nisapi
import polars as pl
import scipy.stats as st
import yaml

import iup
import iup.utils


def preprocess(
    raw_data: pl.LazyFrame,
    filters: dict,
    keep: List[str],
    groups: List[str] | None,
    season_start_month: int,
    season_start_day: int,
) -> iup.CumulativeUptakeData:
    data = iup.CumulativeUptakeData(
        raw_data.filter([pl.col(k).is_in(v) for k, v in filters.items()])
        .select(keep)
        .sort("time_end")
        .collect()
        .with_columns(
            season=pl.col("time_end").pipe(
                iup.utils.date_to_season,
                season_start_month=season_start_month,
                season_start_day=season_start_day,
            ),
            uci=pl.when(pl.col("uci") > pl.col("estimate"))
            .then(pl.col("uci"))
            .otherwise(pl.col("estimate") + 0.0000005),
            lci=pl.when(pl.col("lci") < pl.col("estimate"))
            .then(pl.col("lci"))
            .otherwise(pl.col("estimate") - 0.0000005),
        )
        .with_columns(
            estimate=pl.when(pl.col("estimate") > 0)
            .then(pl.col("estimate"))
            .otherwise(0.0005),
            uci=pl.when(pl.col("estimate") > 0)
            .then(pl.col("uci"))
            .otherwise(pl.col("uci") + 0.0005),
            lci=pl.when(pl.col("estimate") > 0)
            .then(pl.col("lci"))
            .otherwise(pl.col("lci") + 0.0005),
        )
        .with_columns(
            uci_logodds=-(((1 / pl.col("uci")) - 1).log()),
            lci_logodds=-(((1 / pl.col("lci")) - 1).log()),
        )
        .with_columns(logodds=(pl.col("uci_logodds") + pl.col("lci_logodds")) / 2)
        .with_columns(p=(pl.col("logodds").exp()) / (1 + (pl.col("logodds").exp())))
        .with_columns(
            N=1
            / (
                pl.col("p")
                * (1 - pl.col("p"))
                * (
                    (
                        (pl.col("uci_logodds") - pl.col("logodds"))
                        / st.norm.ppf(0.975, 0, 1)
                    )
                    ** 2
                )
            )
        )
        .with_columns(
            sample_sig=(
                (
                    ((pl.col("p") ** 2) * (1 - pl.col("p")))
                    + (((1 - pl.col("p")) ** 2) * pl.col("p"))
                )
                * pl.col("N")
                / (pl.col("N") - 1)
            ).sqrt()
        )
        .with_columns(sdev=pl.col("sample_sig") / (pl.col("N").sqrt()))
        .drop(
            [
                "uci",
                "lci",
                "uci_logodds",
                "lci_logodds",
                "logodds",
                "N",
                "sample_sig",
                "p",
            ]
        )
    )

    if groups is not None:
        assert set(data.columns).issuperset(groups)

    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file")
    p.add_argument(
        "--cache", help="NIS cache directory", default=".cache/nisapi/clean/"
    )
    p.add_argument("--output", help="output parquet file", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_data = nisapi.get_nis(path=args.cache)

    clean_data = preprocess(
        raw_data,
        filters=config["data"]["filters"],
        keep=config["data"]["keep"],
        groups=config["data"]["groups"],
        season_start_month=config["data"]["season_start_month"],
        season_start_day=config["data"]["season_start_day"],
    )

    clean_data.write_parquet(args.output)
