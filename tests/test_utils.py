import polars as pl

import iup
import iup.utils


def test_date_to_season(frame):
    """
    Season is extracted from a column of dates
    """
    frame = frame.with_columns(
        season2=pl.col("time_end").pipe(
            iup.utils.date_to_season, season_start_month=9, season_start_day=1
        )
    )

    assert all(frame["season"] == frame["season2"])


def test_date_to_elapsed_handles_no_season_start(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.utils.date_to_elapsed(pl.col("time_end")).over("geography")
    )

    expected = pl.Series([0.0, 0.0, 7.0, 7.0, 14.0, 14.0, 21.0, 21.0])

    assert (output["elapsed"] == expected).all()


def test_date_to_elapsed_handles_season_start(frame):
    """
    Return the time elapsed since the first date by grouping factor.
    """
    output = frame.sort(["time_end", "geography"]).with_columns(
        elapsed=iup.utils.date_to_elapsed(pl.col("time_end"), 9, 1).over("geography")
    )

    expected = pl.Series(
        [
            121.0,
            121.0,
            128.0,
            128.0,
            135.0,
            135.0,
            142.0,
            142.0,
        ]
    )

    assert (output["elapsed"] == expected).all()
