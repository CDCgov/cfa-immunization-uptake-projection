from datetime import date

import numpy as np
import polars as pl
import pytest

import iup
from iup import eval


@pytest.fixture
def data():
    """
    Mock of observed data.
    """
    return pl.DataFrame(
        {
            "time_end": pl.date_range(
                date(2020, 1, 1), date(2020, 1, 5), interval="1d", eager=True
            ),
            "estimate": [0.0, 0.1, 0.7, 0.4, 0.5],
        }
    )


@pytest.fixture
def pred():
    """
    Mock of point-estimate prediction.
    """
    return pl.DataFrame(
        {
            "time_end": pl.date_range(
                date(2020, 1, 1), date(2020, 1, 5), interval="1d", eager=True
            ),
            "estimate": [0.0, 0.2, 1.0, 0.6, 0.5],
            "quantile": 0.5,
        }
    )


@pytest.fixture
def score_funs():
    """
    Mock of scoring functions.
    """
    return {
        "mspe": eval.mspe,
    }


def test_summarize_score(
    data,
    pred,
    score_funs,
):
    """
    Return the expected forecast start, end and correct MSPE.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.QuantileForecast(pred)

    output = eval.summarize_score(data=data, pred=pred, score_funs=score_funs)

    assert output.item(0, "quantile") == 0.5
    assert output.item(0, "forecast_start") == date(2020, 1, 1)
    assert output.item(0, "forecast_end") == date(2020, 1, 5)
    assert output.item(0, "score_name") == "mspe"
    assert isinstance(output.item(0, "score_value"), float)


def test_mspe():
    """
    Test the mean squared prediction error.
    """
    x = np.array([0.0, 0.1, 0.7, 0.4, 0.5])
    y = np.array([0.0, 0.2, 1.0, 0.6, 0.5])
    assert np.isclose(eval.mspe(x, y), 0.028)


@pytest.fixture
def score_df():
    """
    Mock of the joined data frame between data and prediction
    """
    return pl.DataFrame(
        {
            "time_end": [
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 4),
                date(2020, 1, 5),
            ],
            "data": [0.0, 0.1, 0.7, 0.4, 0.5],
            "pred": [0.0, 0.2, 1.0, 0.6, 0.5],
        }
    )


def test_abs_diff(score_df):
    """
    Test the absolute difference.
    """
    selected_date = date(2020, 1, 1)

    f = eval.abs_diff(selected_date, pl.col("time_end"))

    score_df = score_df.select(score=f(pl.col("data"), pl.col("pred")))

    expected = pl.DataFrame({"score": [0.0, None, None, None, None]})
    assert score_df.equals(expected)
