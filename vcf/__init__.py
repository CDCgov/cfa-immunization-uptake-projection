import os

# silence Jax CPU warning
os.environ["JAX_PLATFORMS"] = "cpu"

import abc
import datetime
import inspect
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing_extensions import Self


def to_season(
    date: pl.Expr,
    season_start_month: int,
    season_end_month: int,
    season_start_day: int = 1,
    season_end_day: int = 1,
) -> pl.Expr:
    """
    Identify the overwinter season from a date.

    Every year, there is a season end (e.g., May 1) and a season start (e.g., Sep 1).
    Dates before the season end are associated with the prior season (e.g., Feb 1, 2020
    belongs to 2019/2020 season). Dates after the season start are associated with the
    next season (e.g., Oct 1, 2020 belongs to 2020/2021). Dates between the season end
    and season start are not in any season (e.g., June 1).

    Args:
        date: dates
        season_start_month: first month
        season_end_month: last month
        season_start_day: first day
        season_end_day: last day

    Returns:
        season like "2020/2021"
    """
    assert (season_start_month, season_start_day) > (
        season_end_month,
        season_end_day,
    ), "Only overwinter seasons are supported"

    # year of this date
    y = date.dt.year()
    # start and end dates of seasons in this year
    end = pl.date(y, season_end_month, season_end_day)
    start = pl.date(y, season_start_month, season_start_day)

    # first year of the two-year season
    sy1 = pl.when(date <= end).then(y - 1).when(date >= start).then(y).otherwise(None)

    return pl.when(sy1.is_null()).then(None).otherwise(pl.format("{}/{}", sy1, sy1 + 1))


def eos_abs_diff(
    obs: pl.DataFrame, pred: pl.DataFrame, features: list[str]
) -> pl.DataFrame:
    """Calculate the absolute difference between observed data and prediction for the last date in a season.

    Args:
        obs: Observed data.
        pred: Predicted data.
        features: Feature column names (must include 'season').

    Returns:
        Data frame with absolute difference scores for end-of-season dates.
    """
    assert "season" in features

    return (
        pred.filter(pl.col("quantile") == 0.5)
        .join(
            obs.filter((pl.col("time_end") == pl.col("time_end").max()).over(features)),
            on=["time_end"] + features,
            how="right",
        )
        .rename({"estimate_right": "obs", "estimate": "pred"})
        .with_columns(
            score_value=(pl.col("pred") - pl.col("obs")).abs(),
            score_fun=pl.lit("eos_abs_diff"),
        )
        .select(features + ["model", "forecast_date", "score_fun", "score_value"])
        .drop_nulls()
    )


class CoverageModel(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        data: pl.DataFrame,
        forecast_date: datetime.date,
        params: dict[str, Any],
        season: dict[str, Any],
        quantiles: list[float],
    ):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self) -> pl.DataFrame:
        pass


class LPLModel(CoverageModel):
    """
    Subclass of CoverageModel for a mixed Logistic Plus Linear model.
    For details, see the online docs.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        forecast_date: datetime.date,
        params: dict[str, Any],
        season: dict[str, Any],
        quantiles: list[float],
        date_column: str = "time_end",
    ):
        """Initialize with a seed and the model structure.

        Args:
            data: Cumulative coverage data for fitting and prediction.
            forecast_date: Date to split fit and prediction data.
            params: All parameters including parameter names and values to specify prior distributions, Control parameters for MCMC fitting, and season start month and day
            quantiles: Posterior sample quantiles
            date_column: Name of the date column in the data. Defaults to "time_end".
        """
        self.raw_data = data
        self.date_column = date_column
        self.quantiles = quantiles
        self.forecast_date = forecast_date
        self.season = season

        # use parameters, separating MCMC and model fitting parameters
        self.params = params

        mcmc_keys = {"num_warmup", "num_samples", "num_chains", "progress_bar"}
        self.mcmc_params = {k: v for k, v in params.items() if k in mcmc_keys}
        self.model_params = {
            k: v
            for k, v in params.items()
            if k in inspect.signature(self._logistic_plus_linear).parameters
        }
        self.fit_key, self.pred_key = random.split(random.key(self.params["seed"]), 2)

        # input data validation
        assert {self.date_column, "estimate", "season", "geography"}.issubset(
            self.raw_data.columns
        )

        # preprocess data
        self.data = (
            self.raw_data
            # prepare observation data
            .with_columns(N_vax=(pl.col("N_tot") * pl.col("estimate")).round(0))
            # add interaction term
            .with_columns(
                season_geo=pl.concat_str(["season", "geography"], separator="_")
            )
            .with_columns(
                t=self._days_in_season(
                    pl.col(date_column),
                    season_start_month=self.season["start_month"],
                    season_start_day=self.season["start_day"],
                )
                / 365
            )
        )

        # set up encoder
        self.features = ("season", "geography", "season_geo")
        self.enc = OrdinalEncoder(dtype=np.int64).fit(
            self.data.select(self.features).to_numpy()
        )
        self.n_feature_levels = [len(x) for x in self.enc.categories_]  # type: ignore

        # initialize MCMC. `None` is a placeholder indicating fitting has not occurred
        self.mcmc = None

    @staticmethod
    def _days_in_season(
        date: pl.Expr, season_start_month: int, season_start_day: int
    ) -> pl.Expr:
        """Extract a time elapsed column from a date column, as polars expressions.

        Args:
            date: Dates
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            number of days elapsed since the first date
        """
        # for every date, figure out the season breakpoint in that year
        season_start = pl.date(date.dt.year(), season_start_month, season_start_day)

        # for dates before the season breakpoint in year, subtract a year
        year = date.dt.year()
        season_start_year = pl.when(date < season_start).then(year - 1).otherwise(year)

        # rewrite the season breakpoints to that immediately before each date
        season_start = pl.date(season_start_year, season_start_month, season_start_day)

        # return the number of days from season start to each date
        return (date - season_start).dt.total_days()

    def model(self, data: pl.DataFrame):
        if "N_vax" in data.columns:
            N_vax = jnp.array(data["N_vax"])
        else:
            N_vax = None

        return self._logistic_plus_linear(
            N_vax=N_vax,
            t=jnp.array(data["t"]),
            # jax runs into a problem if you don't specify this type
            N_tot=jnp.array(data["N_tot"], dtype=jnp.int32),
            feature_levels=jnp.array(
                self.enc.transform(data.select(self.features).to_numpy())
            ),
            **self.model_params,
        )

    def _logistic_plus_linear(
        self,
        N_vax: jnp.ndarray | None,
        t: jnp.ndarray,
        N_tot: jnp.ndarray,
        feature_levels: jnp.ndarray,
        muA_shape1: float,
        muA_shape2: float,
        sigmaA_rate: float,
        tau_shape1: float,
        tau_shape2: float,
        K_shape: float,
        K_rate: float,
        muM_shape: float,
        muM_rate: float,
        sigmaM_rate: float,
        D_shape: float,
        D_rate: float,
    ):
        """
        Logistic Plus Linear model

        Args:
            t: Fraction of a year elapsed since the start of season at each data point.
            N_vax: Number of people vaccinated at each data point, or `None`.
            N_tot: Total number of people in the population at each data point.
            feature_levels: Numeric codes for feature levels: row = data point, col = feature.
            muA_shape1: Beta distribution shape1 parameter for muA prior.
            muA_shape2: Beta distribution shape2 parameter for muA prior.
            sigmaA_rate: Exponential distribution rate parameter for sigmaA prior.
            tau_shape1: Beta distribution shape1 parameter for tau prior.
            tau_shape2: Beta distribution shape2 parameter for tau prior.
            K_shape: Gamma distribution shape parameter for K prior.
            K_rate: Gamma distribution rate parameter for K prior.
            muM_shape: Gamma distribution shape parameter for muM prior.
            muM_rate: Gamma distribution rate parameter for muM prior.
            sigmaM_rate: Exponential distribution rate parameter for sigmaM prior.
            D_shape: Gamma distribution shape parameter for D prior.
            D_rate: Gamma distribution rate parameter for D prior.
        """
        # Sample the overall average value for each parameter
        muA = numpyro.sample("muA", dist.Beta(muA_shape1, muA_shape2))
        muM = numpyro.sample("muM", dist.Gamma(muM_shape, muM_rate))
        tau = numpyro.sample("tau", dist.Beta(tau_shape1, tau_shape2))
        K = numpyro.sample("K", dist.Gamma(K_shape, K_rate))
        D = numpyro.sample("D", dist.Gamma(D_shape, D_rate))

        sigmaA = numpyro.sample(
            "sigmaA", dist.Exponential(sigmaA_rate), sample_shape=(len(self.features),)
        )
        sigmaM = numpyro.sample(
            "sigmaM", dist.Exponential(sigmaM_rate), sample_shape=(len(self.features),)
        )
        zA = numpyro.sample(
            "zA", dist.Normal(0, 1), sample_shape=(sum(self.n_feature_levels),)
        )
        zM = numpyro.sample(
            "zM", dist.Normal(0, 1), sample_shape=(sum(self.n_feature_levels),)
        )

        v = self._vgt(
            t=t,
            feature_levels=feature_levels,
            muA=muA,
            sigmaA=sigmaA,
            zA=zA,
            muM=muM,
            sigmaM=sigmaM,
            zM=zM,
            K=K,
            tau=tau,
        )

        numpyro.sample("obs", dist.BetaBinomial(v * D, (1 - v) * D, N_tot), obs=N_vax)  # type: ignore

    def _vgt(self, t, feature_levels, muA, sigmaA, zA, muM, sigmaM, zM, K, tau):
        """Latent coverage v_g(t)"""
        deltaA = zA * np.repeat(sigmaA, np.array(self.n_feature_levels))
        deltaM = zM * np.repeat(sigmaM, np.array(self.n_feature_levels))

        A = muA + np.sum(deltaA[feature_levels], axis=1)
        M = muM + np.sum(deltaM[feature_levels], axis=1)

        return A / (1 + jnp.exp(-K * (t - tau))) + (M * t)  # type: ignore

    def fit(self) -> Self:
        """Fit a mixed Logistic Plus Linear model on training data.

        A hierarchical model is built with feature-level effects for logistic
        maximum and linear slope, which induce group-specific trajectories.
        Other parameters are non-hierarchical.

        Uses the data, features, model_params, and mcmc_params specified during
        initialization.

        Returns:
            Self with the fitted model stored in the mcmc attribute.
        """
        self.kernel = NUTS(self.model, init_strategy=init_to_sample)
        self.mcmc = MCMC(self.kernel, **self.mcmc_params)
        self.mcmc.run(
            self.fit_key,
            self.data.filter(pl.col(self.date_column) <= self.forecast_date),
        )

        if "progress_bar" in self.mcmc_params and self.mcmc_params["progress_bar"]:
            self.mcmc.print_summary()

        return self

    def predict(self) -> pl.DataFrame:
        """
        Make projections from a fit Logistic Plus Linear model.

        Returns:
            Sample forecast data frame with predictions for dates after forecast_date.
        """

        assert self.mcmc is not None, "Need to fit() first"

        quantile_preds = [
            pl.DataFrame({"quantile": q, "estimate": self._predict_quantile(q)})
            for q in self.quantiles
        ]

        out_data = self.data.select("geography", "season", "time_end").with_columns(
            forecast_date=self.forecast_date
        )

        return pl.concat(
            [pl.concat([out_data, qp], how="horizontal") for qp in quantile_preds],
            how="vertical",
        )

    def _predict_quantile(self, q: float) -> np.ndarray:
        assert self.mcmc is not None
        # pull posterior samples for all parameters (except D) and get the desired quantile
        samples = self.mcmc.get_samples()
        n_samples = len(samples["K"])

        preds = np.stack(
            [
                self._vgt(
                    t=self.data["t"].to_numpy(),
                    feature_levels=self.enc.transform(
                        self.data.select(self.features).to_numpy()
                    ),
                    **{k: v[i,] for k, v in samples.items() if k != "D"},
                )
                for i in range(n_samples)
            ]
        )

        return np.quantile(preds, q=q, axis=0).astype(np.float64)


class RFModel(CoverageModel):
    def __init__(
        self,
        data: pl.DataFrame,
        params: dict[str, Any],
        season: dict[str, Any],
        forecast_date: datetime.date,
        quantiles: list[float],
        date_column: str = "time_end",
    ):
        self.raw_data = data
        self.date_column = date_column
        self.forecast_date = forecast_date
        self.quantiles = quantiles
        self.season = season
        self.params = params

        # other params include max_depth, min_samples_split, min_samples_leaf
        rf_keys = {"n_estimators"}
        self.rf_params = {k: v for k, v in params.items() if k in rf_keys}

        data_t = self.raw_data.with_columns(
            t=pl.col(self.date_column).map_elements(self._month_in_season)
        ).sort(["season", "geography", "t"])

        # preprocessing
        self.date_crosswalk = data_t.select("season", date_column, "t").unique()

        self.data = (
            data_t.select(["season", "geography", "t", "estimate"])
            .pivot(on="t", values="estimate", sort_columns=True)
            .sort(["season", "geography"])
            .pipe(self._impute)
        )

        self.forecast_season = pl.select(
            to_season(
                pl.lit(self.forecast_date),
                season_start_month=self.season["start_month"],
                season_end_month=self.season["end_month"],
                season_end_day=self.season["end_day"],
                season_start_day=self.season["start_day"],
            )
        ).item()
        self.forecast_month = self._month_in_season(self.forecast_date)

    @staticmethod
    def _impute(
        df: pl.DataFrame, index_cols: tuple[str, ...] = ("season", "geography")
    ):
        to_impute_df = df.drop(index_cols)
        imputed_np = KNNImputer(n_neighbors=2).fit_transform(to_impute_df.to_numpy())
        imputed_df = pl.concat(
            [
                df.select(index_cols),
                pl.DataFrame(imputed_np, schema=to_impute_df.columns),
            ],
            how="horizontal",
        )
        assert imputed_df.null_count().sum_horizontal().item() == 0, (
            "Null remaining in data"
        )
        return imputed_df

    def _month_in_season(self, date: datetime.date) -> int:
        assert date.day == 1
        year = date.year
        # start of a season that's in this year
        ssiy = datetime.date(year, self.season["start_month"], self.season["start_day"])

        # season start year
        if date < ssiy:
            ssy = year - 1
        else:
            ssy = year

        return (year - ssy) * 12 + (date.month - self.season["start_month"])

    def fit(self) -> Self:
        self.enc = RFEncoder().fit(self.data)

        self.X_features = ["season", "geography"] + [
            str(t)
            for t in range(0, self.forecast_month + 1)
            if str(t) in self.data.columns
        ]
        self.y_features = [
            str(t)
            for t in range(self.forecast_month + 1, 12)
            if str(t) in self.data.columns
        ]

        # fit the model
        data_fit = self.data.filter(pl.col("season") < self.forecast_season)
        X_fit = self.enc.encode(data_fit.select(self.X_features))
        y_fit = data_fit.select(self.y_features).to_numpy()

        # sklearn complains if you pass a column vector rather than a 1d array
        if y_fit.shape[1] == 1:
            y_fit = y_fit.ravel()

        self.model = RandomForestRegressor(**self.rf_params).fit(X_fit, y_fit)

        return self

    def predict(self) -> pl.DataFrame:
        # make the forecast
        data_pred = self.data.filter(pl.col("season") >= self.forecast_season)

        X_data = data_pred.select(self.X_features)
        assert X_data.shape[0] > 0, f"RF prediction for {self.forecast_date} failed."
        X_pred = self.enc.encode(X_data)

        # make predictions using each tree
        y_tree = np.stack([tree.predict(X_pred) for tree in self.model.estimators_])

        return pl.concat(
            [
                self._postprocess(
                    data_pred=data_pred,
                    y_pred=np.quantile(y_tree, q=q, axis=0),
                    quantile=q,
                )
                for q in self.quantiles
            ]
        )

    def _postprocess(
        self, data_pred: pl.DataFrame, y_pred: np.ndarray, quantile: float
    ) -> pl.DataFrame:
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        return (
            data_pred.select(["season", "geography"])
            .hstack(pl.DataFrame(y_pred, schema=self.y_features))
            .unpivot(
                on=self.y_features,
                index=["season", "geography"],
                variable_name="t",
                value_name="estimate",
            )
            .with_columns(pl.col("t").cast(pl.Int64))
            .join(self.date_crosswalk, on=["season", "t"], how="left")
            .drop("t")
            .with_columns(forecast_date=self.forecast_date, quantile=quantile)
        )


class RFEncoder:
    def __init__(self, categorical_features: tuple = ("season", "geography")):
        self.categorical_features = categorical_features
        self.enc = OneHotEncoder(sparse_output=False)

    def fit(self, data: pl.DataFrame) -> Self:
        self.enc.fit(data.select(self.categorical_features).to_numpy())
        return self

    def encode(self, data: pl.DataFrame) -> np.ndarray:
        X_enc = self.enc.transform(data.select(self.categorical_features).to_numpy())
        X_pass = data.drop(self.categorical_features).to_numpy()

        assert isinstance(X_enc, np.ndarray)
        return np.asarray(np.hstack((X_enc, X_pass)))
