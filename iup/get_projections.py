import sys
import os
import polars as pl
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime as dt
import re


def z_scale(
        nat_data,
        old_data
):
    """
    Standardize data using z-score given the mean and SD of user-defined data. 

    Parameters:
    --------------
    nat_data: polars.Series / 1-column polars.DataFrame / numpy.array
        The data at the natural scale to be standardized
    old_data: polars.Series / 1-column polars.DataFrame / numpy.array
        The user-defined data whose mean and SD are used for z-score standardization

    Return:
    -------------
    A polars.Series with z-score standardized data
    
    """
    
    if isinstance(nat_data, pl.DataFrame):
        if(nat_data.shape[1] != 1):
            raise ValueError("Raw data must be 1 column.")
        else:
            nat_data = nat_data.to_series()
    elif not isinstance(nat_data, (pl.Series,np.ndarray)):
        raise ValueError("Raw data type is" + type(nat_data) + ', not pl.Series or numpy array.')
    
    if isinstance(old_data, pl.DataFrame):
        if(old_data.shape[1] != 1):
            raise ValueError("Data used to scale must be 1 column.")
    elif not isinstance(old_data, (pl.Series,np.ndarray)):
        raise ValueError("Data used to scale's type is" + type(old_data) + ', not pl.Series or numpy array.')

    z_data = (nat_data - old_data.mean())/old_data.std()

    return z_data



def inv_z_scale(
        z_data, 
        old_data
):
    """
    Inverse transformation from the z-score standardized data 
    back to natural scale given the mean and SD of user-defined data. 

    Parameters:
    --------------
    z_data: polars.Series / 1-column polars.DataFrame / numpy.array
        The z-score standardized data to be inversely transformed
    old_data: polars.Series / 1-column polars.DataFrame / numpy.array
        The user-defined data whose mean and SD are used for inverse transformation

    Return:
    -------------
    A polars.Series with data at natural scale 

    """

    if isinstance(z_data, pl.DataFrame):
        if(z_data.shape[1] != 1):
            raise ValueError("Scaled data must be 1 column.")
        else:
            z_data = z_data.to_series()
    elif not isinstance(z_data, (pl.Series,np.ndarray)):
        raise ValueError("Scaled data type is" + type(z_data) + ', not pl.Series or numpy array.')
    
    if isinstance(old_data, pl.DataFrame):
        if(old_data.shape[1] != 1):
            raise ValueError("Data used to scale must be 1 column.")
    elif not isinstance(old_data, (pl.Series,np.ndarray)):
        raise ValueError("Data used to scale's type is" + type(old_data) + ', not pl.Series or numpy array.')

    nat_data = z_data * old_data.std() + old_data.mean()
    
    return nat_data


# real-time projection 
def get_projections(
        train_data, 
        data_to_initiate, 
        end_date,
        max_day_length = 10
):
    
    """
    Use linear regression model to iteratively predict the vaccine uptake
    given an initialization date. 
    
    Parameters: 
    --------------------
    train_data: polars.DataFrame
        A pl.DataFrame used for training linear regression. Should be the vaccine uptake
        data from the entire previous season. Must include 
        the uptake from the previous day (`previous`) and the number of days from 
        vaccine rollout date (`elapsed`) at the natural scale. 
    data_to_initiate: polars.DataFrame
        A pl.DataFrame used as the start of prediction. Must include `date`, 
        the uptake from the previous day (`previous`) and the number of days from 
        vaccine rollout date (`elapsed`) at the natural scale.
    end_date: None / string with %Y-%m-%d format / dt.date object / dt.datetime object
        The end date of vaccine uptake projection. If None, use the last date in the 
        `data_to_initiate`.
    max_day_length: integer
        The maximum day length allowed between the vaccine rollout date and the 
        first day when the vaccine data became available in the previous season.

    Descriptions:
    --------------------
    This function uses the vaccine uptake data from the previous season to predict the 
    uptake in the next season, starting at a user-defined date.  

    Returns
    --------------------
    A polars.DataFrame with 1-day interval of predicted vaccine uptake. 
    
    """

    
    # check data format of end_date #
    if end_date is None:
        end_date = data_to_initiate.select('date')[len(data_to_initiate)-1][0]
    elif isinstance(end_date,str):
        if re.match(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$',end_date):
            end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        else: 
            raise ValueError("The format of end_date is not %Y-%m-%d.")
    elif isinstance(end_date,dt.date) | isinstance(end_date, dt.datetime):
        end_date = end_date
    else:
        raise ValueError("The format of end_date is neither date, datetime, or string.")

    # should we check for others? #
    if train_data[0,'elapsed'] > max_day_length :
        train_data = train_data.slice(1)

    # data preparation: z-scale #
    scaled_train = train_data.with_columns(
        previous = z_scale(train_data['previous'],train_data['previous']),
        elapsed = z_scale(train_data['elapsed'],train_data['elapsed']),
        daily = z_scale(train_data['daily'],train_data['daily'])
        )

    scaled_train = scaled_train.drop_nulls()

    scaled_x = scaled_train.with_columns(
            interact = pl.col('previous') * pl.col('elapsed')
        ).select("previous", "elapsed", "interact").to_numpy()

    scaled_y = scaled_train.select('daily').to_numpy()

    # linear regression #
    model = LinearRegression().fit(scaled_x, scaled_y)
   
    # Sequential prediction # 
    start_date = data_to_initiate.select('date')[0]
    date_series = pl.date_range(start_date, end_date, '1d',eager = True) 
    # date_series = date_series.filter(date_series.is_in(data_to_initiate['date']))

    raw_data = data_to_initiate
    pred_data = pl.DataFrame()

    for date in date_series[1:len(date_series)+1]:

        raw_data =  raw_data.with_columns(
                state = pl.lit('US').alias('state'),  # NOTE: pl.lit is an expression for literal value
                elapsed = pl.col('elapsed') + (date - raw_data['date'][0]).days,
                date = pl.lit(date),
                previous = pl.col('daily'),
                daily = None
            )


        scaled_x = raw_data.select("previous", "elapsed").with_columns(
                    previous = z_scale(raw_data['previous'],train_data['previous']),
                    elapsed = z_scale(raw_data['elapsed'],train_data['elapsed'])
                    ).with_columns(
                    interact = pl.col('previous') * pl.col('elapsed')
                ).to_numpy()

        scaled_y = model.predict(scaled_x)[0]

        # scale the predicted daily using the daily from train data #
        fitted_daily = inv_z_scale(scaled_y, train_data['daily'])

        raw_data = raw_data.with_columns(
                daily = pl.lit(fitted_daily)
        ).with_columns(
            cumulative = pl.col('cumulative') + pl.col('previous')
        )

        pred_data = pl.concat([pred_data, 
                            raw_data],
                                how = 'vertical')

    pred_data = pred_data.select(
        'state','date','elapsed','cumulative','daily','previous'
    )
    return(pred_data)
    

