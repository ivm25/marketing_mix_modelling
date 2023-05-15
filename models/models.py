# IMPORT ALL THE LIBRARIES


import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm


simulated_data_df = pd.read_csv("data/de_simulated_data.csv")

# Wrangle Data ----

simulated_data_df.info()

simulated_data_df.columns = simulated_data_df.columns.str.lower()

simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])



# Add/Remove Features for Modeling ----

correlation_df = simulated_data_df\
    .assign(date_month = lambda x:x['date'].dt.month_name())\
        .drop(["date","search_clicks_p","facebook_i"], axis = 1)





# STEP 2.0 MODELING WITH ADSTOCK ----

# Basic Adstock Model ----

def adstock(series, rate):
    """_summary_

    Args:
        series (_type_): _description_
        rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    tt = np.empty(len(series))
    for i in range(1, len(series)):
        tt[i] = series[i] + series[i-1]*rate
        
    tt_series = pd.Series(tt, index = series.index)
    
    tt_series.name = "adstock_" + str(series.name)
    return tt_series


# Model using statsmodel


def model_summary_to_dataframe(model):
    # Extract the middle table
    model_results_df = pd.DataFrame(model.summary().tables[1])
    # set the index
    model_results_df = model_results_df.set_index(0)
    
    model_results_df.columns = model_results_df.iloc[0]
    
    model_results_df = model_results_df.iloc[1:]
    
    model_results_df.index.name = 'Parameter'
    
    # fetch the surrounding information within the summary table
    
    metrics_on_top = pd.DataFrame(model.summary().tables[0])
    metrics_1 = metrics_on_top[[0,1]] 
    metrics_2 = metrics_on_top[[2,3]]
    metrics_2.columns = 0,1
    metrics_top_out = pd.concat([metrics_1, metrics_2])
    metrics_top_out.columns = ['Parameter', 'Value']
    
    metrics_at_bottom = pd.DataFrame(model.summary().tables[2])
    metrics_bot_1 = metrics_at_bottom[[0,1]] 
    metrics_bot_2 = metrics_at_bottom[[2,3]]
    metrics_bot_2.columns = 0,1
    metrics_bot_out = pd.concat([metrics_bot_1, metrics_bot_2])
    metrics_bot_out.columns = ['Parameter', 'Value']
    
    metrics_df = pd.concat([metrics_top_out, metrics_bot_out])
    
    return pd.DataFrame(model_results_df), metrics_df
    
    