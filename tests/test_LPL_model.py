import datetime

import numpyro.infer
import polars as pl
import polars.testing

import iup
import iup.models

PARAMS = {
    "seed": 42,
    "muA_shape1": 100.0,
    "muA_shape2": 140.0,
    "sigmaA_rate": 40.0,
    "tau_shape1": 100.0,
    "tau_shape2": 225.0,
    "K_shape": 20.0,
    "K_rate": 5.0,
    "muM_shape": 1.0,
    "muM_rate": 0.1,
    "sigmaM_rate": 40,
    "D_shape": 5.0,
    "D_rate": 0.01,
    "num_warmup": 10,
    "num_samples": 10,
    "num_chains": 1,
    "progress_bar": False,
}
SEASON = {
    "start_month": 7,  # July
    "start_day": 1,
    "end_month": 4,  # April
    "end_day": 1,
    "start_year": 2018,
    "end_year": 2022,
}

QUANTILES = [0.025, 0.5, 0.975]


def test_index():
    df = pl.DataFrame(
        {
            "first_name": ["John", "John", "Eve", "Eve"],
            "last_name": ["Smith", "Adams", "Fulani", "Kumar"],
            "height": [1.1, 2.2, 3.3, 4.4],
        }
    )

    out = iup.models.LPLModel._index(df, groups=["first_name", "last_name"])
    print(out)

    polars.testing.assert_frame_equal(
        out,
        df.with_columns(
            pl.Series("first_name_idx", [1, 1, 0, 0]),
            pl.Series("last_name_idx", [3, 0, 1, 2]),
        ),
    )


def test_preprocess(frame):
    """
    Should produce expected columns, given raw data.
    """
    data = iup.models.LPLModel._preprocess(
        data=frame,
        date_column="time_end",
        season_start_month=SEASON["start_month"],
        season_start_day=SEASON["start_day"],
    )

    expected_cols = {
        "geography",
        "time_end",
        "estimate",
        "lci",
        "uci",
        "N_tot",
        "season",
        "N_vax",
        "season_geo",
        "elapsed",
        "season_idx",
        "geography_idx",
        "season_geo_idx",
    }

    assert expected_cols == set(data.columns)


def test_fit_handles_groups(frame):
    """
    Model should produce posterior samples for each parameter.
    """

    model = iup.models.LPLModel(
        data=frame,
        forecast_date=datetime.date(2020, 1, 1),
        season=SEASON,
        params=PARAMS,
        quantiles=QUANTILES,
    )

    model.fit()
    assert isinstance(model.mcmc, numpyro.infer.MCMC)

    dimensions = [value.shape[0] for _, value in model.mcmc.get_samples().items()]
    assert all(d == 10 for d in dimensions)
