
# %%
# Set up the package path 

import sys
import os
import polars as pl
from sklearn.linear_model import LinearRegression
sys.path.append('/home/ab32/github/cfa-immunization-uptake-projection/iup')

# %%
# load modules

from get_uptake_data import get_uptake_data
from build_projection_model import build_projection_model
from make_projections import make_projections

os.chdir('..')
# os.chdir('cfa-immunization-uptake-projection')

# %%
# misc functions #
# data_prepare function including drop_first and scaling #

nis_usa_2022.head()

# data = nis_usa_2022

# %%

# ask Ed why it's deciding on the z-scale value instead of the value of elapsed.import polars as pl
def z_scale(nat_data,old_data):
    
    # z-scale #
    
    # data must be 1-column pl.DataFrame or a pl.Series #
    if isinstance(nat_data, pl.DataFrame):
        if(nat_data.shape[1] != 1):
            raise ValueError("Raw data must be 1 column.")
        else:
            nat_data = nat_data.to_series()
    elif not isinstance(nat_data, pl.Series):
        raise ValueError("Raw data type is" + type(nat_data) + ', not pl.Series.')
    
    if isinstance(old_data, pl.DataFrame):
        if(old_data.shape[1] != 1):
            raise ValueError("Data used to scale must be 1 column.")
    elif not isinstance(old_data, pl.Series):
        raise ValueError("Data used to scale's type is" + type(old_data) + ', not pl.Series.')

    z_data = (nat_data - old_data.mean())/old_data.std()

    return z_data



# %%
# inverse z-scale #

def inv_z_scale(z_data, old_data):
    
    # inverse z scale #
    
    # data must be 1-column pl.DataFrame or a pl.Series #
    if isinstance(z_data, pl.DataFrame):
        if(z_data.shape[1] != 1):
            raise ValueError("Scaled data must be 1 column.")
        else:
            z_data = z_data.to_series()
    elif not isinstance(z_data, pl.Series):
        raise ValueError("Scaled data type is" + type(z_data) + ', not pl.Series.')
    
    if isinstance(old_data, pl.DataFrame):
        if(old_data.shape[1] != 1):
            raise ValueError("Data used to scale must be 1 column.")
    elif not isinstance(old_data, pl.Series):
        raise ValueError("Data used to scale's type is" + type(old_data) + ', not pl.Series.')

    nat_data = z_data * old_data.std() + old_data.mean()
    
    return nat_data
    

# %%

# real-time projection 
def get_real_time_projection(train_data, start_date, end_date,
                             max_day_length = 10):
    
    # data preparation: drop first #

    # if the date of the first row is away from the rollout date
    # more than 'max_day_length', remove the first row
    if(train_data[0,'elapsed'] > max_day_length):
        train_data = train_data.slice(1)
    
    # data preparation: z-scale #
    # NOTE: add for loop in future optimize #
    train_data = train_data.with_columns(
    previous = z_scale(train_data['previous'],train_data['previous']),
    elapsed = z_scale(train_data['elapsed'],train_data['elapsed']),
    daily = z_scale(train_data['daily'],train_data['daily'])
    )

        
# model building only on the prepared data #



# %%
