import polars as pl
import abc
import typing


class Data(pl.DataFrame, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def validate(self) -> None:
        pass


class IncidentUptakeData(Data):
    def validate(self):
        # must have columns: start_date, end_date, estimate
        assert set(self.columns).issuperset({"start_date", "end_date", "estimate"})
        # columns must have expected types
        assert all(self[x].dtype.is_temporal() for x in ["start_date", "end_date"])
        assert self["estimate"].dtype.is_numeric()


class CumulativeUptakeData(Data):
    def validate(self):
        # must have columns: date, estimate
        assert set(self.columns).issuperset({"date", "estimate"})
        # columns must have expected data types
        assert self["date"].dtype.is_temporal()
        assert self["estimate"].dtype.is_numeric()
        # cumulative uptakes must be nonnegative
        assert (self["estimate"] >= 0.0).all()

    def to_incident(self) -> IncidentUptakeData:
        # use the dates and relevant start and end dates
        # take the diff to get estimates
        # number of output rows should be input rows minus one
        raise NotImplementedError

    def interpolate(self, dates: pl.Series, method: str = "linear") -> typing.Self:
        # given some dates, linearly interpolate at the new dates
        # do some check that method=="linear"; this gives you an opening to use
        #   more clever interpolations if you need them later
        # check that the output has the same number of rows as the input
        # validate before returning!
        raise NotImplementedError


class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: Data) -> typing.Self:
        pass

    @abc.abstractmethod
    def predict(self, data: Data, *args, **kwargs) -> Data:
        pass


def LinearCumulativeUptakeModel(Model):
    def __init__(self):
        # probably wants self.model = scikit learn regression model
        pass

    def fit(self, data: CumulativeUptakeData) -> typing.Self:
        # check that we have the kind of data this model can accept
        assert isinstance(data, CumulativeUptakeData)
        data.validate()

        # convert dates to time, pass that and estimates to scikit learn
        # regression object
        # self.model.fit(time, estimate)
        raise NotImplementedError
        return self

    def predict(self, dates):
        # convert dates to time? -- this might need to be a class or package
        #  method, to make sure you're doing the date->time and time->date
        #  conversions all the same way
        # then use the scikit learn objects .predict() method
        # return those results
        raise NotImplementedError

def evaluate(model, score, training_data, eval_data) -> float:
    # - fit the model on the training data
    # - predict over the dates in the eval data
    # - score the prediction vs. actual eval data
    # - return that
    # unclear if this should be an Evaluator object that takes multiple models
    #   and scores

def get_nis(path) -> CumulativeUptakeData:
    # pull from eg https://data.cdc.gov/Flu-Vaccinations/Weekly-Cumulative-Influenza-Vaccination-Coverage-A/2v3t-r3np/about_data
    # maybe using pl.read_csv so you immediately get a pl.DataFrame
    # do all the cleaning and filtration
    # when you're ready, wrap the output: out = CumulativeUptakeData(my_clean_df)
    # and validate it: out.validate()
    # before returning: return out
    raise NotImplementedError
