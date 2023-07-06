import pandas as pd
import numpy as np
import pathlib
from joblib import load
import json


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm


from models.models import adstock

# Data load

simulated_data_df = pd.read_csv("data/de_simulated_data.csv")

simulated_data_df.columns = simulated_data_df.columns.str.lower()
simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])   
 
correlation_df = simulated_data_df\
    .assign(date_month = lambda x:x['date'].dt.month_name())\
        .drop(["date","search_clicks_p","facebook_i"], axis = 1)


def generate_grid(size = 100):
        
    rng = np.random.default_rng(123)
    size = 100
    budget_grid_df = pd.DataFrame(dict(
    adstock_tv_s = rng.uniform(0, 1, size = size),
    adstock_ooh_s = rng.uniform(0 , 1, size = size),
    adstock_print_s = rng.uniform(0 , 1, size = size),
    adstock_search_s = rng.uniform(0, 1, size  = size),
    adstock_facebook_s = rng.uniform(0, 1, size = size)
        
    ))
    return budget_grid_df

grid = generate_grid(size = 100)


def adstock_model(correlation_df,
                   grid,
                   ):
    
    
    model_details = {}
    
    for tv, ooh, prnt, search, facebook in zip(grid.adstock_tv_s,
                                                grid.adstock_ooh_s,
                                                grid.adstock_print_s,
                                                grid.adstock_search_s,
                                                grid.adstock_facebook_s):
        
        adstock_tv_new = adstock(correlation_df.tv_s,tv)
        adstock_ooh_new = adstock(correlation_df.ooh_s, ooh)
        adstock_facebook_new = adstock(correlation_df.facebook_s, facebook)
        adstock_print_new = adstock(correlation_df.print_s, prnt)
        adstock_search_new = adstock(correlation_df.search_s, search)
        
       
        dep_vars = pd.concat([adstock_tv_new, 
                              adstock_ooh_new, 
                              adstock_print_new,
                              adstock_search_new,
                              adstock_facebook_new],
                              axis = 1)
    
        ind_vars = correlation_df.revenue
        
        x_adstock_train, x_adstock_test, y_adstock_train, y_adstock_test = train_test_split(dep_vars,
                                                            ind_vars,
                                                            random_state = 0)
        
        cons = sm.add_constant(x_adstock_train)

        model_ols = sm.OLS(y_adstock_train, x_adstock_train).fit()
        
        if tv not in model_details:
            model_details[tv] = model_ols
        print(model_details)
        
    return model_details
        
def extract_coefficients(model_dict = None):
    dataframe_dict = {}
    for key in model_dict:
        
        model = model_dict.get(key)
        # print(model.summary())
        models_data = pd.DataFrame(model.summary().tables[1])\
                        .iloc[1:6,0:2]\
                            .rename(columns = {0:'variable', 1:'coefficient'})
        
        models_data['coefficient'] = models_data['coefficient'].astype(str)
        models_data['coefficient'] = models_data['coefficient'].astype(float)
        models_data['variable'] = models_data['variable'].astype(str)
        models_data['key variable'] = models_data['variable'].str[8:]
        print(models_data)
        if key not in dataframe_dict:
            dataframe_dict[key] = models_data
    return dataframe_dict






