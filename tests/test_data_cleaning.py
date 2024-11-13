import iup
from datetime import date


def test_apply_filters_handles_filters():
    """
    If multiple filters are given to apply_filters, all of them should be applied.
    """
    frame = iup.IncidentUptakeData(
        {
            "geography": ["USA", "PA", "USA"],
            "date": [date(2020, 1, 7), date(2020, 1, 14), date(2020, 1, 21)],
            "estimate": [0.0, 1.0, 2.0],
            "indicator": ["refused", "booster", "booster"],
        }
    )

    filters = {"geography": "USA", "indicator": "booster"}

    output = iup.apply_filters(frame, filters)

    assert output.shape[0] == 1
    assert output["estimate"][0] == 2.0


def test_apply_filters_handles_no_filters():
    """
    If no filters are given to apply_filters, the whole frame is returned.
    """
    frame = iup.IncidentUptakeData(
        {
            "geography": ["USA", "PA", "USA"],
            "date": [date(2020, 1, 7), date(2020, 1, 14), date(2020, 1, 21)],
            "estimate": [0.0, 1.0, 2.0],
            "indicator": ["refused", "booster", "booster"],
        }
    )

    filters = None

    output = iup.apply_filters(frame, filters)

    assert output.shape[0] == 3
