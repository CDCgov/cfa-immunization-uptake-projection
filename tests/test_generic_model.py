import polars as pl

import iup.models


def test_extract_group_combos_handles_groups(frame):
    """
    Find all unique grouping factor combos.
    """
    frame = iup.models.extract_group_combos(frame, ["geography", "season"])

    assert isinstance(frame, pl.DataFrame)
    assert frame.shape[0] == 2


def test_extract_group_combos_handles_no_groups(frame):
    """
    Returns none since no groups are declared.
    """
    frame = iup.models.extract_group_combos(frame, None)

    assert frame is None
