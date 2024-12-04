import polars as pl
import iup
import pytest
from datetime import date


@pytest.fixture
def data():
    """
    Mock of observed data.
    """
    data = pl.DataFrame(
        {
            "date": pl.date_range(
                date(2020, 1, 1), date(2020, 1, 5), interval="1d", eager=True
            ),
            "estimate": [0.0, 0.1, 0.7, 0.4, 0.5],
        }
    )

    return data


@pytest.fixture
def pred():
    """
    Mock of point-estimate prediction.
    """
    pred = pl.DataFrame(
        {
            "date": pl.date_range(
                date(2020, 1, 1), date(2020, 1, 5), interval="1d", eager=True
            ),
            "estimate": [0.0, 0.2, 1.0, 0.6, 0.5],
        }
    )

    return pred


def test_get_mspe(data, pred):
    """
    Return the expected forecast start, end and correct MSPE.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.PointForecast(pred)

    output = iup.get_mspe(data, pred)

    output_date = output.select("forecast_start", "forecast_end")

    assert output_date.equals(
        pl.DataFrame(
            {"forecast_start": date(2020, 1, 1), "forecast_end": date(2020, 1, 5)}
        )
    )

    assert np.isclose(output["mspe"][0], 0.028)


def test_get_mean_bias(data, pred):
    """
    Return the expected forecast start, end and correct mean bias.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.PointForecast(pred)

    output = iup.get_mean_bias(data, pred)

    output_date = output.select("forecast_start", "forecast_end")

    assert output_date.equals(
        pl.DataFrame(
            {"forecast_start": date(2020, 1, 1), "forecast_end": date(2020, 1, 5)}
        )
    )

    assert abs(output["mbias"][0] - (-0.6)) < 1e-6


def test_get_eos_abe(data, pred):
    """
    Return the expected forecast start, end and correct end-of-season error%.
    """
    data = iup.IncidentUptakeData(data)
    pred = iup.PointForecast(pred)

    output = iup.get_eos_abe(data, pred)

    output_date = output.select("forecast_start", "forecast_end")

    assert output_date.equals(
        pl.DataFrame(
            {"forecast_start": date(2020, 1, 1), "forecast_end": date(2020, 1, 5)}
        )
    )

    assert abs(output["ae_prop"][0] - (0.352941)) < 1e-6
