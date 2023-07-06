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

import pathlib
from joblib import load
import json
# PATHS

# BASE_PATH =pathlib.Path(__file__).parent.resolve()

# ART_PATH = BASE_PATH.joinpath("artefacts").resolve()

# # ARTEFACTS

# best_model = load(ART_PATH.joinpath("best_model"))

# X_Train = load(ART_PATH.joinpath("X_train"))

# media_spend_df = load(ART_PATH.joinpath("media_spend_df"))



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
    


# FUNCTIONS (make them modular later on)

# def generate_grid(size = 100):
        
#     rng = np.random.default_rng(123)
#     size = 100
#     budget_grid_df = pd.DataFrame(dict(
#         adstock_tv_s = rng.uniform(0, 1, size = size),
#         adstock_ooh_s = rng.uniform(0 , 1, size = size),
#         adstock_print_s = rng.uniform(0 , 1, size = size),
#         adstock_search_s = rng.uniform(0, 1, size  = size),
#         adstock_facebook_s = rng.uniform(0, 1, size = size)
        
#     ))
#     return budget_grid_df



# def optimize_budget(df, 
#                     grid, 
#                     media_spend,
#                     verbose = True):
    
#     X_train = df
#     budget_grid_df = grid
#     media_spend_df = media_spend
    
#     best_budget = dict(
#         rebalancing_coef = None,
#         media_spend_rebal = None,
#         score = None
#     )
    
#     for i , row in enumerate(budget_grid_df.index):
        
#         # scale the randome budget mix
#         budget_scaled = budget_grid_df.loc[i,:]/np.sum(budget_grid_df.loc[i,:])
        
#         # create rebalancing coefficients
        
#         rebalancing_coef_df = budget_scaled\
#             .to_frame()\
#                 .reset_index()\
#                     .set_axis(['name','value'], axis = 1)
       
#     # Rebalance adstock
    
#     X_train_adstock = X_train[X_train.columns[X_train.columns.str.startswith('adstock_')]] 
    
#     X_train_not_adstock = X_train[X_train.columns[~X_train.columns.str.startswith('adstock_')]]
    
#     X_train_adstock_rebal = X_train_adstock.copy()
    
#     X_train_adstock_rebal.sum(axis = 1)
    
    
#     for i, col in enumerate(X_train_adstock_rebal.columns):
#         X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]
        
    
#     X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)
    
#     # Make Predictions
#     predicted_revenue_current = best_model['model'][0].predict(X_train).sum()
    
#     predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()
    
#     score = predicted_revenue_new/predicted_revenue_current
    
#     # Media spend rebalanced
    
#     total_current_spend = media_spend_df['spend_current'].sum()
    
#     media_spend_rebalanced_df = media_spend_df.copy()
    
#     media_spend_rebalanced_df['spend_new'] = total_current_spend * (rebalancing_coef_df['value'])
    
#     if best_budget['score'] is None or score > best_budget['score']:
#         best_budget['rebalancing_coef'] = rebalancing_coef_df,
#         best_budget['media_spend_rebal'] = media_spend_rebalanced_df
#         best_budget['score'] = score
        
#         if verbose:
#             print("New Best Budget:")
#             print(best_budget)
#             print("\n")
            
#     if verbose:
#         print("Done!")
    
#     return best_budget


# def adstock_search(df, 
#                    grid, 
#                    verbose = False):
    
#     best_model = dict(
#         model = None,
#         params = None,
#         score = None,
#     )
    
#     for tv, ooh, prnt, search, facebook in zip(grid.adstock_tv_s,
#                                                 grid.adstock_facebook_s,
#                                                 grid.adstock_ooh_s,
#                                                 grid.adstock_print_s,
#                                                 grid.adstock_search_s):
#         adstock_tv_s_new = adstock(df.tv_s,tv)
#         adstock_ooh_s_new = adstock(df.ooh_s, ooh)
#         adstock_print_s_new = adstock(df.print_s, prnt)
#         adstock_facebook_s_new = adstock(df.facebook_s, facebook)
#         adstock_search_s_new = adstock(df.search_s, search)
        
#         x_new = pd.concat([adstock_tv_s_new, 
#                        adstock_ooh_s_new, 
#                        adstock_print_s_new,
#                        adstock_search_s_new,
#                        adstock_facebook_s_new,
#                        df.competitor_sales_b,
#                        pd.get_dummies(df.date_month)],
#                        axis = 1)
        
#         y_new = df.revenue
        
#         x_adstock_train, x_adstock_test, y_adstock_train, y_adstock_test = train_test_split(x_new,
#                                                             y_new,
#                                                             random_state = 0)
        
#         new_model = Pipeline([('lm2', LinearRegression())])
        
#         new_model.fit(x_adstock_train, y_adstock_train)
        
#         score = new_model.score(x_adstock_test, y_adstock_test)
        
#         if best_model['model'] is None or score > best_model['score']:
#             best_model['model'] = new_model,
#             best_model['params'] = dict(tv = tv,
#                                         ooh = ooh,
#                                         prnt = prnt,
#                                         search = search,
#                                         facebook = facebook)
#             best_model['score'] = score
            
#             if verbose:
#                 print("New Best Model:")
#                 print(best_model)
#                 print("\n")
                
#     if verbose:
#         print("Done")
    
#     return best_model    
