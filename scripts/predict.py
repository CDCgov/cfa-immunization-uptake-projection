import argparse
import pickle
from pathlib import Path

import polars as pl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True)
    p.add_argument("--output", help="forecasts parquet", required=True)
    args = p.parse_args()

    with open(args.fits, "rb") as f:
        models_dict = pickle.load(f)

    forecasts = pl.concat(
        [
            model.predict().with_columns(model=pl.lit(model.__class__.__name__))
            for model in models_dict.values()
        ]
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    forecasts.write_parquet(args.output)
