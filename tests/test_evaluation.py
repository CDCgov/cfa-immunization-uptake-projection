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


def test_score_df(data, pred):
    """
    Return the expected forecast start, end and correct MSPE.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.PointForecast(pred)

    output = eval.score(data=data, pred=pred, score_fun=eval.mspe)
    assert output.item(0, "forecast_start") == date(2020, 1, 1)
    assert output.item(0, "forecast_end") == date(2020, 1, 5)
    # we're not testing the actual value, just that we get some value
    assert isinstance(output["score"][0], float)


def test_mspe():
    x = np.array([0.0, 0.1, 0.7, 0.4, 0.5])
    y = np.array([0.0, 0.2, 1.0, 0.6, 0.5])
    assert np.isclose(eval.mspe(x, y), 0.028)


def test_mean_bias(data, pred):
    """
    Return the expected forecast start, end and correct mean bias.
    """
    x = pl.Series([0.0, 0.1, 0.7, 0.4, 0.5])
    y = pl.Series([0.0, 0.2, 1.0, 0.6, 0.5])
    assert np.isclose(eval.mean_bias(x, y), -0.6)


def test_eos_abe(data, pred):
    """
    Return the expected forecast start, end and correct end-of-season error%.
    """
    x = pl.Series([0.0, 0.1, 0.7, 0.4, 0.5])
    y = pl.Series([0.0, 0.2, 1.0, 0.6, 0.5])
    assert np.isclose(eval.eos_abe(x, y), 0.352941)
