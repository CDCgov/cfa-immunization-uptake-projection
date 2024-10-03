
# %%
# Set up the package path 

import sys
import os
import polars as pl
sys.path.append('/home/ab32/github/cfa-immunization-uptake-projection/iup')

# %%
# load modules

from get_uptake_data import get_uptake_data
from build_projection_model import build_projection_model
from make_projections import make_projections

# os.chdir('..')
os.chdir('cfa-immunization-uptake-projection')

# %%
# misc functions #
# data_prepare function including drop_first and scaling #

nis_usa_2022.head()

data = nis_usa_2022

# %%
# Data prepare: drop_first and z-scale 

# include the decision of dropping the first or not: ask Ed why it's deciding on the z-scale value instead of the value of elapsed.import polars as pl
def data_prepare(data,
                 old_data,
                 max_day_length = 10):
    
    # if the date of the first row is away from the rollout date
    # more than 'max_day_length', remove the first row
    if(data[0,'elapsed'] > max_day_length):
        data = data.slice(1)
    
    # z-scale #







# model building only on the prepared data #

# %%
