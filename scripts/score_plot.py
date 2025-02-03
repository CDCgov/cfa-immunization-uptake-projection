import argparse

import altair as alt
import polars as pl
import yaml


def plot_score(scores):
    """
    Save a evaluation score plot, varied by forecast start date.

    Arguments:
    --------------
    scores: polars.DataFrame
        The evaluation scores data frame.
    config: dict
        config.yaml to define the saving path and the saving name.

    Return:
    -------------
    None. The graph is saved.

    """

    scores = scores.collect()
    score_names = scores["score_fun"].unique()

    score_dict = config["score_dict"]

    charts = []
    for score_name in score_names:
        score = scores.filter(pl.col("score_fun") == score_name)

        score_chart = (
            alt.Chart(score)
            .mark_point()
            .encode(
                alt.X("forecast_start:T", title="Forecast Start"),
                alt.Y("score:Q", title="Score"),
            )
            .properties(title=score_dict[score_name])
        )

        charts = charts + [score_chart]

    return alt.hconcat(*charts).configure_title(fontSize=30)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="config file", default="scripts/config.yaml")
    p.add_argument("--score", help="evaluation scores")
    p.add_argument("--output", help="output png file")
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    score = pl.scan_parquet(args.score).collect()

    if config["score_plot"]["plot"]:
        plot_score(score).save(args.output)
