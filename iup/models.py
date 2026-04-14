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
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import Self

import iup
from iup.utils import date_to_season, month_order


class CoverageModel(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        data: pl.DataFrame,
        forecast_date: datetime.date,
        params: dict[str, Any],
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
        data: iup.CumulativeCoverageData,
        forecast_date: datetime.date,
        params: dict[str, Any],
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
        self.data = self._preprocess(
            data=self.raw_data, date_column=self.date_column, params=self.params
        )

        # do the indexing
        self.n_group_levels = [
            self.data.select(pl.col(group).unique()).height
            for group in ["season", "geography", "season_geo"]
        ]

        # initialize MCMC. `None` is a placeholder indicating fitting has not occurred
        self.mcmc = None

    @classmethod
    def _preprocess(cls, data: pl.DataFrame, date_column, params) -> pl.DataFrame:
        out = (
            data
            # prepare observation data
            .rename({"sample_size": "N_tot"})
            .with_columns(N_vax=(pl.col("N_tot") * pl.col("estimate")).round(0))
            # add interaction term
            .with_columns(
                season_geo=pl.concat_str(["season", "geography"], separator="_")
            )
            .with_columns(
                t=cls._days_in_season(
                    pl.col(date_column),
                    params["start_month"],
                    params["start_day"],
                )
            )
            .with_columns(elapsed=pl.col("t") / 365)
        )

        # add the indices
        out = cls._index(out, groups=["season", "geography", "season_geo"])

        return out

    @staticmethod
    def _index(data: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
        """
        For each column in `groups` (e.g., `"season"`), add a new column `"{group}_idx"`
        (e.g., `"season_idx"`) that has the values in the original column replaced by
        integer indices.

        Args:
            data: dataframe
            groups: names of columns

        Returns: dataframe with additional columns like `"{group}_idx"`
        """
        for group in groups:
            unique_values = (
                data.select(pl.col(group).unique().sort()).get_column(group).to_list()
            )
            indices = list(range(len(unique_values)))
            replace_map = {value: index for value, index in zip(unique_values, indices)}
            data = data.with_columns(
                pl.col(group).replace_strict(replace_map).alias(f"{group}_idx")
            )

        return data

    @staticmethod
    def _days_in_season(
        date_col: pl.Expr, season_start_month: int, season_start_day: int
    ) -> pl.Expr:
        """Extract a time elapsed column from a date column, as polars expressions.

        Args:
            date_col: Column of dates.
            season_start_month: First month of the overwinter disease season.
            season_start_day: First day of the first month of the overwinter disease season.

        Returns:
            number of days elapsed since the first date
        """
        # for every date, figure out the season breakpoint in that year
        season_start = pl.date(date_col.dt.year(), season_start_month, season_start_day)

        # for dates before the season breakpoint in year, subtract a year
        year = date_col.dt.year()
        season_start_year = (
            pl.when(date_col < season_start).then(year - 1).otherwise(year)
        )

        # rewrite the season breakpoints to that immediately before each date
        season_start = pl.date(season_start_year, season_start_month, season_start_day)

        # return the number of days from season start to each date
        return (date_col - season_start).dt.total_days()

    def model(self, data: pl.DataFrame):
        # missing `N_vax` column signals that we are drawing predictions, not fitting
        if "N_vax" in data.columns:
            N_vax = jnp.array(data["N_vax"])
        else:
            N_vax = None

        return self._logistic_plus_linear(
            N_vax=N_vax,
            elapsed=jnp.array(data["elapsed"]),
            # jax runs into a problem if you don't specify this type
            N_tot=jnp.array(data["N_tot"], dtype=jnp.int32),
            groups=jnp.array(
                data.select(
                    [f"{group}_idx" for group in ["season", "geography", "season_geo"]]
                )
            ),
            n_groups=3,
            n_group_levels=self.n_group_levels,
            **self.model_params,
        )

    @staticmethod
    def _logistic_plus_linear(
        N_vax: jnp.ndarray | None,
        elapsed: jnp.ndarray,
        N_tot: jnp.ndarray,
        groups: jnp.ndarray,
        n_groups: int,
        n_group_levels: list[int],
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
        """Fit a mixed Logistic Plus Linear model on training data.

        Args:
            elapsed: Fraction of a year elapsed since the start of season at each data point.
            N_vax: Number of people vaccinated at each data point, or `None`.
            N_tot: Total number of people in the population at each data point.
            groups: Numeric codes for groups: row = data point, col = grouping factor.
            n_groups: Number of grouping factors.
            n_group_levels: Number of unique levels of each grouping factor.
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
        D = numpyro.sample("d", dist.Gamma(D_shape, D_rate))

        sigmaA = numpyro.sample(
            "sigmaA", dist.Exponential(sigmaA_rate), sample_shape=(n_groups,)
        )
        sigmaM = numpyro.sample(
            "sigmaM", dist.Exponential(sigmaM_rate), sample_shape=(n_groups,)
        )
        zA = numpyro.sample(
            "zA", dist.Normal(0, 1), sample_shape=(sum(n_group_levels),)
        )
        zM = numpyro.sample(
            "zM", dist.Normal(0, 1), sample_shape=(sum(n_group_levels),)
        )
        deltaA = zA * np.repeat(sigmaA, np.array(n_group_levels))
        deltaM = zM * np.repeat(sigmaM, np.array(n_group_levels))

        A = muA + np.sum(deltaA[groups], axis=1)
        M = muM + np.sum(deltaM[groups], axis=1)

        # Calculate latent true coverage at each datum
        v = A / (1 + jnp.exp(-K * (elapsed - tau))) + (M * elapsed)  # type: ignore

        numpyro.sample("obs", dist.BetaBinomial(v * D, (1 - v) * D, N_tot), obs=N_vax)  # type: ignore

    def fit(self) -> Self:
        """Fit a mixed Logistic Plus Linear model on training data.

        If grouping factors are specified, a hierarchical model will be built with
        group-specific parameters for the logistic maximum and linear slope,
        drawn from a shared distribution. Other parameters are non-hierarchical.

        Uses the data, groups, model_params, and mcmc_params specified during
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

        assert self.mcmc is not None, f"Need to fit() first; mcmc is {self.mcmc}"

        predictive = Predictive(self.model, self.mcmc.get_samples())
        # run the predictions, not using the observations
        pred = predictive(self.pred_key, self.data.drop("N_vax"))

        # observations are rows; posterior samples are columns
        pred = np.array(pred["obs"]).transpose()

        # put predictions into a dataframe
        sample_cols = [f"_sample_{i}" for i in range(pred.shape[1])]
        pred = pl.DataFrame(pred, schema=sample_cols)

        index_cols = [self.date_column, "N_tot", "season", "geography", "season_geo"]

        data_pred = (
            pl.concat([self.data, pred], how="horizontal")
            .unpivot(
                on=sample_cols,
                index=index_cols,
                variable_name="sample_id",
                value_name="estimate",
            )
            .with_columns(
                forecast_date=self.forecast_date,
                # convert from sample_id strings to integers
                sample_id=pl.col("sample_id")
                .replace_strict({name: i for i, name in enumerate(sample_cols)})
                .cast(pl.UInt64),
                estimate=pl.col("estimate") / pl.col("N_tot"),
            )
            .group_by(
                ["season", "geography", "season_geo", "time_end", "forecast_date"]
            )
            .agg(
                quantile=pl.concat_arr(self.quantiles),
                estimate=pl.concat_arr(
                    [pl.quantile("estimate", q).alias(str(q)) for q in self.quantiles]
                ),
            )
        )

        return iup.QuantileForecast(data_pred.explode(["quantile", "estimate"]))


class RFModel(CoverageModel):
    def __init__(
        self,
        data: iup.CumulativeCoverageData,
        params: dict[str, Any],
        forecast_date: datetime.date,
        quantiles: list[float],
        date_column: str = "time_end",
    ):
        self.raw_data = data
        self.date_column = date_column
        self.forecast_date = forecast_date
        self.quantiles = quantiles
        self.params = params
        self.months = month_order(self.params["start_month"])
        self.end_month_index = self.months.index(
            datetime.date(
                self.params["end_year"],
                self.params["end_month"],
                self.params["end_day"],
            ).strftime("%b")
        )

        rf_keys = {"n_estimators"}
        # other params include max_depth, min_samples_split, min_samples_leaf, can be added later #

        self.rf_params = {k: v for k, v in params.items() if k in rf_keys}

        self.data = self._preprocess(
            self.raw_data,
            self.months,
            self.end_month_index,
            self.date_column,
        )

    @classmethod
    def _preprocess(
        cls, data: pl.DataFrame, months, end_month_index, date_column
    ) -> pl.DataFrame:
        out = (
            data.with_columns(
                t=pl.col(date_column)
                .dt.to_string("%b")
                .map_elements(lambda x: months.index(x) - end_month_index, pl.Int64)
            )
            .filter(pl.col("t").is_between(1 - end_month_index, 0))
            .select(["season", "geography", "t", "estimate"])
            .with_columns(pl.format("t={}", pl.col("t")))
            .pivot(on="t", values="estimate")
            .drop_nulls()
            .sort(["season", "geography"])
        )

        return out

    def fit(self) -> Self:
        self.enc = CoverageEncoder()
        self.enc.fit(self.data)

        target_season = date_to_season(
            pl.lit(self.forecast_date),
            season_start_month=self.params["start_month"],
            season_start_day=self.params["start_day"],
        )

        forecast_t = (
            self.months.index(self.forecast_date.strftime("%b")) - self.end_month_index
        )

        end_date = datetime.date(
            self.params["end_year"], self.params["end_month"], self.params["end_day"]
        )

        # this is true only when target_season is the last season in the data, which is our case for now
        data_fit = self.data.filter(pl.col("season") != target_season)

        self.models = {}

        ## fit all the data after forecast_t ##
        for target_t in range(forecast_t + 1, 1):
            features = ["season", "geography"] + [
                f"t={t}"
                for t in range(
                    1 - self.months.index(end_date.strftime("%b")), forecast_t + 1
                )
            ]

            # always fit the last month of the season given the months before forecast_date
            X_fit = self.enc.encode(data_fit.select(features))
            y_fit = data_fit.select(f"t={target_t}").to_series().to_numpy()

            self.models[target_t] = RandomForestRegressor(**self.rf_params).fit(
                X_fit, y_fit
            )

        return self

    def predict(self) -> pl.DataFrame:
        data_pred = self.data.filter(
            pl.col("season")
            == date_to_season(
                pl.lit(self.forecast_date),
                season_start_month=self.params["start_month"],
                season_start_day=self.params["start_day"],
            )
        )

        forecast_t = (
            self.months.index(self.forecast_date.strftime("%b")) - self.end_month_index
        )

        end_date = datetime.date(
            self.params["end_year"], self.params["end_month"], self.params["end_day"]
        )

        features = ["season", "geography"] + [
            f"t={t}"
            for t in range(
                1 - self.months.index(end_date.strftime("%b")), forecast_t + 1
            )
        ]

        X_pred = self.enc.encode(data_pred.select(features))
        quantile_cols = [f"q={k}" for k in self.quantiles]
        index_cols = ["season", "geography"]

        preds = []

        for target_t, model in self.models.items():
            pred = np.array([tree.predict(X_pred) for tree in model.estimators_])
            pred = {f"q={k}": np.quantile(pred, k, axis=0) for k in self.quantiles}
            pred = pl.DataFrame(pred)

            full_pred = (
                pl.concat(
                    [data_pred.select(["season", "geography"]), pred], how="horizontal"
                )
                .unpivot(
                    on=quantile_cols,
                    index=index_cols,
                    variable_name="quantile",
                    value_name="estimate",
                )
                .with_columns(
                    pl.col("quantile").str.replace("q=", "").cast(pl.Float64),
                    forecast_date=self.forecast_date,
                    target_index=(
                        target_t + self.end_month_index
                    ),  # convert back to month index
                    target_year=pl.col("season").str.extract(r"^(\d{4})/\d{4}"),
                )
                .with_columns(
                    season_start_date=pl.date(
                        pl.col("target_year"),
                        self.params["start_month"],
                        self.params["start_day"],
                    ),
                    target_index=pl.col("target_index").cast(pl.String) + "mo",
                )
                .with_columns(
                    pl.col("season_start_date")
                    .dt.offset_by(pl.col("target_index"))
                    .alias("time_end")
                )
                .drop(["target_index", "target_year", "season_start_date"])
            )

            preds.append(full_pred)

        return pl.concat(preds)


class CoverageEncoder:
    def __init__(self, categorical_feature_names: tuple = ("season", "geography")):
        self.categorical_feature_names = categorical_feature_names
        self.enc = OneHotEncoder(sparse_output=False)
        self.categorical_features = None

    def fit(self, data: pl.DataFrame):
        self.enc.fit(data.select(self.categorical_feature_names).to_numpy())

        self.categorical_features = list(
            self._iter_features(self.categorical_feature_names, self.enc.categories_)
        )

    @staticmethod
    def _iter_features(names, categories):
        for feature, values in zip(names, categories):
            for value in values:
                yield (feature, value)

    def encode(self, data: pl.DataFrame) -> np.ndarray:
        X_enc = self.enc.transform(
            data.select(self.categorical_feature_names).to_numpy()
        )
        X_pass = data.drop(self.categorical_feature_names).to_numpy()

        assert isinstance(X_enc, np.ndarray)
        return np.asarray(np.hstack((X_enc, X_pass)))

    def categories(self, data: pl.DataFrame):
        if self.categorical_features is None:
            raise RuntimeError
        else:
            return self.categorical_features + [
                ("unencoded", col)
                for col in data.drop(self.categorical_feature_names).columns
            ]
