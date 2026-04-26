import polars as pl
import pytest

from vcf import to_season


def test_before():
    assert (
        pl.select(
            to_season(pl.date(2020, 1, 1), season_end_month=4, season_start_month=7)
        ).item()
        == "2019/2020"
    )


def test_after():
    assert (
        pl.select(
            to_season(pl.date(2020, 11, 1), season_end_month=4, season_start_month=7)
        ).item()
        == "2020/2021"
    )


def test_no_season():
    assert (
        pl.select(
            to_season(pl.date(2020, 6, 1), season_end_month=4, season_start_month=7)
        ).item()
        is None
    )


def test_overwinter():
    with pytest.raises(AssertionError):
        pl.select(
            to_season(pl.date(2020, 6, 1), season_end_month=7, season_start_month=4)
        )
