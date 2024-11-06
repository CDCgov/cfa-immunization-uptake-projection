import polars as pl
import altair as alt

# get mspe #
def get_mspe(
        data_df,
        pred_df,
        var
):
    """
    Calculate the mean squared prediction error between the observed
    data and prediction when both are available. 

    Parameters:
    --------------
    data_df: polars.DataFrame
        The observed vaccine uptake. Must include `date` and the variable-
        of-interest (var).
    pred_df: polars.DataFrame
        The predicted daily uptake. Must include `date` and the variable-
        of-interest (var).
    var: string
        The name of the variable-of-interest.
    
    Return:
    -------------

    A polars.DataFrame with the MSPE and the date of initializing forecast. 
    
    """

    if not any(data_df['date'].is_in(pred_df['date'])):
        ValueError('No matched dates between data_df and pred_df.')

    common_dates = data_df.filter(
        pl.col('date').is_in(pred_df['date'])
    ).select('date')

    if len(common_dates)!= common_dates.n_unique():
        ValueError('Duplicated dates are found in data_df or pred_df.')
    
    
    mspe = data_df.join(
            pred_df, 
            on = 'date',
            how = 'inner'
        ).with_columns(
            spe = (pl.col(var) - pl.col(f"{var}_right"))**2
        ).with_columns(
            mspe = pl.col('spe').mean(),
        ).filter(
            pl.col('date') == pl.col('date').min(),
        ).select(
            'date','mspe'
        )
    

    return mspe


# get end-of-season total uptake #

def get_eos(
        pred_df,
        var = 'cumulative'
):
    """

    Get the `var` on the last date in the `pred_df`. In this function, 
    it is to get the predicted total uptake at the end of the season.
    
    Parameters:
    --------------

    pred_df: polars.DataFrame
        The predicted daily uptake. Must include `date` and the variable-
        of-interest (var).
    var: string
        The name of the variable-of-interest.
    
    Return:
    -------------
    A polars.DataFrame with the variable-of-interest (end-of-season uptake)
    
    on the last date and the last date. 
    
    """
    
    eos = pred_df.filter(
        pl.col('date') == pl.col('date').max()
    ).rename(
        {'date':'end_date'}
    ).select(var,'end_date')
    
    return eos

# plot projections #
def plot_projections(
        obs_data,
        pred_data_list,
        n_columns,
        pic_loc,
        pic_name
):

    """
    Save a multiple-grid graph with the comparison between the observed uptake and the prediction,
    initiated over the season.

    Parameters:
    --------------
    obs_data: polars.Dataframe
        The observed uptake with `date` and `cumulative`, indicating the cumulative vaccine uptake as of `date`.
    pred_data_list: list containing multiple polars.Dataframe
        The predicted daily uptake, differed by initialized date, must include columns `date` and `cumulative`.
    n_columns:
        The number of columns in the graph.
    pic_loc:
        The directory to save the graph.
    pic_name:
        The name of the graph.

    Return:
    -------------
    None. The graph is saved.

    """

    # input check #
    if 'date' not in obs_data.columns or 'cumulative' not in obs_data.columns:
        ValueError("'date' or 'cumulative' is missing from obs_data.")

    if not isinstance(pred_data_list,list):
        ValueError("pred_data_list must be a list.")

    if 'date' not in pred_data_list[0].columns or 'cumulative' not in pred_data_list[0].columns:
        ValueError("'date' or 'cumulative' is missing from pred_data_list.")

    # plot weekly initiated prediction #
    time_axis = alt.Axis(
    format = '%Y-%m',tickCount='month'
    )

    obs = alt.Chart(obs_data).mark_circle(color = 'black').encode(
        alt.X('date',axis = time_axis),
        alt.Y('cumulative')
    )

    charts = []

    dates = [pred_data_list[i]['date'].min() for i in range(len(pred_data_list))]

    for i in range(len(pred_data_list)):

        date = dates[i].strftime('%Y-%m-%d')
        pred = pred_data_list[i]

        chart = alt.Chart(pred).mark_line(color='red').encode(
            x=alt.X('date:T',title='Date',axis=time_axis),
            y=alt.Y('cumulative:Q',title='Cumulative Vaccine Uptake (%)')
        ).properties(
            title = f'Start Date: {date}'
        )

        obs = obs.encode(
            x = chart.encoding.x
        )

        chart = obs + chart
        chart = chart.resolve_scale(
            x ='shared'
        )
        charts.append(chart)


    rows = []

    for i in range(0, len(charts),n_columns):
        row = alt.hconcat(*list(charts[i:i+n_columns]))
        rows.append(row)

    alt.vconcat(
        *rows
    ).configure_title(
        fontSize=30
    ).save(f'{pic_loc}{pic_name}')
