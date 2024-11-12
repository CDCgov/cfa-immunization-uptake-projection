import abc
import polars as pl
from abc import abstractmethod

#### prediction output ####
class Forecast(pl.DataFrame, metaclass=abc.ABCMeta):
    """
    Abstract class for all the forecast data type.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.validate()

    @abc.abstractmethod
    def validate(self) -> None:
        pass


class PointForecast(Forecast):
    """
    Class for forecast with point estimate
    Can be a special case for QuantileForecast.

    """

    def __init__(self):

        super().__init__()
        self.validate()

    # Must have response variable and date #
    # TODO: make the response variable to be a variable not hardcoded #
    def validate(self):

        self.assert_columns_found(["date", "estimate"])
        self.assert_columns_type(["date"], pl.Date)
        self.assert_columns_type(["estimate"], pl.Float64)

    def assert_columns_found(self, columns: list[str]):
        """
        Verify that expected columns are found.

        Parameters
        columns: List[str]
            names of expected columns
        """
        for col in columns:
            assert col in self.columns, f"Column {col} is expected but not found."

    def assert_columns_type(self, columns: list[str], dtype):
        """
        Verify that columns have the expected type.

        Parameters
        columns: List[str]
            names of columns for which to check type
        dtype:
            data type for each listed column
        """
        for col in columns:
            assert (
                self[col].dtype == dtype
            ), f"Column {col} should be {dtype} but is {self[col].dtype}"


class QuantileForecast(Forecast):
    """
    Class for forecast with quantiles.
    Save for future.
    """
    def validate(self) -> None:
        pass

class PostSampleForecast(Forecast):
    """
    Class for forecast with posterior distribution.
    Save for future.
    """
    def validate(self) -> None:
        pass


#### evaluation metrics #####
class PointMetric(pl.DataFrame, metaclass=abc.ABCMeta):

    """
    Abstract class for evaluation metrics for point estimate forecast.
    """
    def __init__(
            self,
            data: PointForecast,
            pred: PointForecast
        ):
        super().__init__(self.preprocess(data, pred))

    """
    Combine data and prediction together, varied by metric type (time-wise or not)
    """
    @abstractmethod
    def preprocess(self, data, pred):
        pass

    """
    Evaluation metric, varied by metric type.
    """
    @abstractmethod
    def get_metric(self, metric_type):
        pass


# Any metric that does not require time-wise matching, can go here.
class TimelessPointMetric(PointMetric):

    def preprocess(self, data, pred):
        pass

    def get_metric(self, metric_type):
        pass


class TimewisePointMetric(PointMetric):

    def __init__(
            self,
            data: PointForecast,
            pred: PointForecast
    ):
        self.validate(data,pred)
        super().__init__(self.preprocess(data,pred))

    def validate(self, data, pred):
        self.assert_date_match(data, pred)

    """
    Check the conditions for date match:
    1. Mutual dates must exist between data and prediction.
    2. There should not be any duplicated date in either data or prediction.
    """

    def assert_date_match(self, data, pred):

        assert any(data['date'].is_in(pred['date'])), 'No matched dates between data and prediction.'

        common_dates = data.filter(
            pl.col('date').is_in(pred['date'])
        ).select('date')

        assert len(common_dates) == common_dates.n_unique(), 'Duplicated dates are found in data or prediction.'

    def preprocess(self, data, pred):
        """
        Join data and prediction with 1:1 validate
        """
        return data.join(
            pred,
            on = 'date',
            how = 'inner',
            validate = '1:1'
        )

    # polymorphism: same function conduct different functionalities based on different argument (metric_type)
    def get_metric(self, metric_type):
        """
        Calculate metric based on `metric_type` of
        joined dataframe from data and prediction.
        """
        if metric_type == 'mspe':
            return self.get_mspe()
        elif metric_type == 'mean_bias':
            return self.get_mean_bias()
        elif metric_type == 'eos_abe':
            return self.get_eos_abe()
        else:
            raise Exception(f'Does not support {metric_type}')

    # metric can be directly called too
    def get_mspe(self):
        """
        Calculate MSPE from joined data
        ----------------------
        Input: self (joined data)
        Return: pl.DataFrame with MSPE and the forecast start date

        """
        start = self.with_columns(
            spe = (pl.col('estimate') - pl.col("estimate_right"))**2
            ).with_columns(
                mspe = pl.col('spe').mean(),
            ).filter(
                pl.col('date') == pl.col('date').min(),
            ).rename(
                {'date':'forecast_start'}
            ).select(
                'forecast_start','mspe'
            )

        end = self.filter(
            pl.col('date') == pl.col('date').max()
        ).rename(
            {'date':'forecast_end'}
        ).select(
            'forecast_end'
        )

        return pl.concat([start,end],how ='horizontal')

    def get_mean_bias(self):
        """
        Calculate Mean bias from joined data.
        Note the bias here is not the classical bias calculated from the posterior distribution.

        The bias here is defined as: at time t,
        bias = -1 if pred_t < data_t; bias = 0 if pred_t == data_t; bias = 1 if pred_t > bias_t

        mean_bias = sum of the bias across time/length of data
        -------------------------
        Input: self (joined data)
        Return: pl.DataFrame with mean bias and the forecast start date
        """
        joined = self.with_columns(
                pl.when(pl.col('estimate') < pl.col('estimate_right')
                ).then(-1)
                .when(pl.col('estimate') == pl.col('estimate_right'))
                .then(0)
                .otherwise(1).alias('bias')
            )

        m_bias = pl.DataFrame({
                           'mbias':joined['bias'].sum()/joined.shape[0],
                           'forecast_start':joined['date'].min(),
                           'forecast_end':joined['date'].max()
                           })

        return m_bias

    def get_eos_abe(self):

        """
        Calculate the absolute error of the total uptake at the end of season between data and prediction.
        Maybe can belong to TimelessPointMetric because not every date needs to be matched,
        but the situation when the last date does not match needs to be defined.
        -------------------
        Input: self (joined data)
        Return: pl.DataFrame with absolute error in the total uptake between data and prediction
                and the forecast end date.
        """
        joined = self.with_columns(
                cumu_data = pl.col('estimate').cum_sum(),
                cumu_pred = pl.col('estimate_right').cum_sum()
            ).filter(
                pl.col('date') == pl.col('date').max()
            ).rename(
                {'date':'forecast_end'}
            )

        abe_perc = abs(joined['cumu_data'] - joined['cumu_pred'])/joined['cumu_data']

        return pl.DataFrame([joined['forecast_end'],abe_perc])


## Draft for quantile and posterior distribution metrics in the future ##
class QuantileMetric(pl.DataFrame, metaclass=abc.ABCMeta):

    def __init__(
            data: QuantileForecast,
            pred: QuantileForecast
        ):
        QuantileMetric.validate(data, pred)
        super().__init__()

    @staticmethod
    def validate(data, pred):
        pass


class PostSampleMetric(pl.DataFrame, metaclass=abc.ABCMeta):

    def __init__(
            data: PostSampleForecast,
            pred: PostSampleForecast
        ):
        PostSampleMetric.validate(data, pred)
        super().__init__()

    @staticmethod
    def validate(data, pred):
        pass
