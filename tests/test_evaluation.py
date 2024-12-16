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


@pytest.mark.parametrize("score_fun", [eval.mspe, eval.mean_bias, eval.eos_abe])
def test_score_df(data, pred, score_fun):
    """
    Return the expected forecast start, end, and correct MSPE.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.PointForecast(pred)

    output = eval.score(data=data, pred=pred, score_fun=score_fun)

    # Validate forecast start and end dates
    assert output.item(0, "forecast_start") == date(2020, 1, 1)
    assert output.item(0, "forecast_end") == date(2020, 1, 5)

    # Validate score type
    assert isinstance(output["score"][0], float)

    # Expected values for different scoring functions
    expected_scores = {eval.mspe: 0.028, eval.mean_bias: -0.6, eval.eos_abe: 0.352941}

    if score_fun in expected_scores:
        assert np.isclose(output["score"][0], expected_scores[score_fun])
