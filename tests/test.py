# %%
import matplotlib as plt
import polars as pl


# %%

    # data preparation: drop first #

    # if the date of the first row is away from the rollout date
    # more than 'max_day_length', remove the first row

    


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

    # data preparation: z-scale #
    # NOTE: add for loop in future optimize #
train_data.with_columns(
    previous = z_scale(train_data['previous'],train_data['previous']))




    
train_data['elapsed'] = z_scale(train_data['elapsed'],
                                train_data['elapsed'])
    
train_data['daily'] = z_scale(train_data['daily'], 
                              train_data['daily'])
    
    # %%