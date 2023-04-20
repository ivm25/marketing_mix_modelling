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



# STEP 1.0 DATA UNDERSTANDING ----

# Wrangle Data ----

simulated_data_df.info()

simulated_data_df.columns = simulated_data_df.columns.str.lower()

simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])



# Visualize Time Series Data ----
simulated_data_df\
    .melt(id_vars = "date")\
        .pipe(px.line,
              x = "date",
              y = "value",
              color = "variable",
              facet_col = "variable",
              facet_col_wrap = 3,
              template = "plotly_dark")\
                  .update_yaxes(matches = None)



# Visualize Current Spend Profile ----
media_spend_df = simulated_data_df\
    .filter(regex = '_s$', axis = 1)\
        .sum()\
            .to_frame()\
                .reset_index()\
                    .set_axis(['media','spend_current'], axis = 1)
                    
                    
media_spend_df\
    .pipe(px.bar,
          x = 'media',
          y = 'spend_current',
          color = 'media',
          template = "plotly_dark"
          )

# Add/Remove Features for Modeling ----

correlation_df = simulated_data_df\
    .assign(date_month = lambda x:x['date'].dt.month_name())\
        .drop(["date","search_clicks_p","facebook_i"], axis = 1)


# Correlations ----

correlation_df\
    .corr()\
        .pipe(sns.heatmap,
              annot =True)




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


adstock_tv_s = adstock(correlation_df.tv_s, 0.5)
adstock_ooh_s = adstock(correlation_df.ooh_s, 0.5)

adstock_print_s = adstock(correlation_df.print_s, 0.5)
adstock_search_s = adstock(correlation_df.search_s, 0.5)

adstock_facebook_s = adstock(correlation_df.facebook_s, 0.5)



# prep fpr modelling

X = pd.concat([adstock_tv_s, 
               adstock_ooh_s, 
               adstock_print_s,
               adstock_search_s,
               adstock_facebook_s,
               correlation_df.competitor_sales_b,
               pd.get_dummies(correlation_df.date_month)
               ], axis = 1)

Y = correlation_df.revenue

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    random_state = 0)


# Model using scikitlearn

pipeline_model = Pipeline([
                     ('lm', LinearRegression())]
)

pipeline_model.fit(x_train,y_train)

pipeline_model.score(x_train, y_train)

y_pred_test = pipeline_model.predict(x_test)

# Score Predictions on Test Set
r2_score(y_test, y_pred_test)

mean_absolute_error(y_test, y_pred_test)
np.sqrt(mean_absolute_error(y_test,y_pred_test))

# Model using statsmodel
cons = sm.add_constant(x_train)

model_ols = sm.OLS(y_train, x_train).fit()

print(model_ols.summary())
# Coefficients to the model

pipeline_model['lm'].coef_
pipeline_model['lm'].intercept_



# Search Adstock Rate (Optimize for Model Fit)

rng = np.random.default_rng(123)

size = 100

max_adstock = 1

adstock_grid_df = pd.DataFrame(dict(
    adstock_tv_s = rng.uniform(0,max_adstock, size = size),
    adstock_ooh_s = rng.uniform(0,max_adstock, size = size),
    adstock_print_s = rng.uniform(0,max_adstock, size = size),
    adstock_search_s = rng.uniform(0,max_adstock, size = size),
    adstock_facebook_s = rng.uniform(0,max_adstock, size = size)
))



# creating adstock search model

def adstock_search(df, 
                   grid, 
                   verbose = False):
    
    best_model = dict(
        model = None,
        params = None,
        score = None,
    )
    
    for tv, ooh, prnt, search, facebook in zip(grid.adstock_tv_s,
                                                grid.adstock_facebook_s,
                                                grid.adstock_ooh_s,
                                                grid.adstock_print_s,
                                                grid.adstock_search_s):
        adstock_tv_s_new = adstock(df.tv_s,tv)
        adstock_ooh_s_new = adstock(df.ooh_s, ooh)
        adstock_print_s_new = adstock(df.print_s, prnt)
        adstock_facebook_s_new = adstock(df.facebook_s, facebook)
        adstock_search_s_new = adstock(df.search_s, search)
        
        x_new = pd.concat([adstock_tv_s_new, 
                       adstock_ooh_s_new, 
                       adstock_print_s_new,
                       adstock_search_s_new,
                       adstock_facebook_s_new,
                       df.competitor_sales_b,
                       pd.get_dummies(df.date_month)],
                       axis = 1)
        
        y_new = df.revenue
        
        x_adstock_train, x_adstock_test, y_adstock_train, y_adstock_test = train_test_split(x_new,
                                                            y_new,
                                                            random_state = 0)
        
        new_model = Pipeline([('lm2', LinearRegression())])
        
        new_model.fit(x_adstock_train, y_adstock_train)
        
        score = new_model.score(x_adstock_test, y_adstock_test)
        
        if best_model['model'] is None or score > best_model['score']:
            best_model['model'] = new_model,
            best_model['params'] = dict(tv = tv,
                                        ooh = ooh,
                                        prnt = prnt,
                                        search = search,
                                        facebook = facebook)
            best_model['score'] = score
            
            if verbose:
                print("New Best Model:")
                print(best_model)
                print("\n")
                
    if verbose:
        print("Done")
    
    return best_model    


best_model = adstock_search(correlation_df[0:100],
                            adstock_grid_df, 
                            verbose = True)



# STEP 3.0: BUDGET REBALANCE ----

best_model_coef = best_model['model'][0]['lm2'].coef_
best_model_coef_names = X.columns

rebalancing_coef_df = pd.DataFrame(dict(name = best_model_coef_names,
                                        value = best_model_coef))\
                                            .set_index('name')\
                                                .filter(regex = "_s$",
                                                        axis = 0)\
                                                            .assign(value = lambda x:x['value']/np.sum(x['value']))\
                                                                .reset_index()
                                                                
                                                                
# Media spend rebabalanced

total_current_spend = media_spend_df["spend_current"].sum()

media_spend_rebalanced_df = media_spend_df.copy()

media_spend_rebalanced_df["spend_new"] = total_current_spend * (rebalancing_coef_df['value'])


media_spend_rebalanced_df\
    .melt(id_vars = 'media')\
        .pipe(px.bar,
              x = 'media',
              y = 'value',
              color = 'media',
              barmode = 'group',
              template = 'plotly_dark'
              )

    
# Predcted Revenue after rebalancing

X_train_adstock = x_train[x_train.columns[x_train.columns.str.startswith('adstock_')]]

X_train_not_adstock = x_train[x_train.columns[~x_train.columns.str.startswith('adstock_')]]

X_train_adstock_rebal = X_train_adstock.copy()

X_train_adstock_rebal.sum(axis = 1)

for i, col in enumerate(X_train_adstock_rebal.columns):
    X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]


X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)

predicted_revenue_current = best_model['model'][0].predict(x_train).sum()

predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()

predicted_revenue_new / predicted_revenue_current


# Step 4 Budget Optimisation

rng = np.random.default_rng(123)
size = 100
budget_grid_df = pd.DataFrame(dict(
    adstock_tv_s = rng.uniform(0, 1, size = size),
    adstock_ooh_s = rng.uniform(0 , 1, size = size),
    adstock_print_s = rng.uniform(0 , 1, size = size),
    adstock_search_s = rng.uniform(0, 1, size  = size),
    adstock_facebook_s = rng.uniform(0, 1, size = size)
    
))


budget_grid_df

for i, row in enumerate(budget_grid_df.index):
    print(budget_grid_df.loc[i,:]/np.sum(budget_grid_df.loc[i,:]))
    


def optimize_budget(df, 
                    grid, 
                    media_spend,
                    verbose = True):
    
    X_train = df
    budget_grid_df = grid
    media_spend_df = media_spend
    
    best_budget = dict(
        rebalancing_coef = None,
        media_spend_rebal = None,
        score = None
    )
    
    for i , row in enumerate(budget_grid_df.index):
        
        # scale the randome budget mix
        budget_scaled = budget_grid_df.loc[i,:]/np.sum(budget_grid_df.loc[i,:])
        
        # create rebalancing coefficients
        
        rebalancing_coef_df = budget_scaled\
            .to_frame()\
                .reset_index()\
                    .set_axis(['name','value'], axis = 1)
       
    # Rebalance adstock
    
    X_train_adstock = X_train[X_train.columns[X_train.columns.str.startswith('adstock_')]] 
    
    X_train_not_adstock = X_train[X_train.columns[~X_train.columns.str.startswith('adstock_')]]
    
    X_train_adstock_rebal = X_train_adstock.copy()
    
    X_train_adstock_rebal.sum(axis = 1)
    
    
    for i, col in enumerate(X_train_adstock_rebal.columns):
        X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]
        
    
    X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)
    
    # Make Predictions
    predicted_revenue_current = best_model['model'][0].predict(X_train).sum()
    
    predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()
    
    score = predicted_revenue_new/predicted_revenue_current
    
    # Media spend rebalanced
    
    total_current_spend = media_spend_df['spend_current'].sum()
    
    media_spend_rebalanced_df = media_spend_df.copy()
    
    media_spend_rebalanced_df['spend_new'] = total_current_spend * (rebalancing_coef_df['value'])
    
    if best_budget['score'] is None or score > best_budget['score']:
        best_budget['rebalancing_coef'] = rebalancing_coef_df,
        best_budget['media_spend_rebal'] = media_spend_rebalanced_df
        best_budget['score'] = score
        
        if verbose:
            print("New Best Budget:")
            print(best_budget)
            print("\n")
            
    if verbose:
        print("Done!")
    
    return best_budget
          
                    
budget_optimized = optimize_budget(
    df          = x_train,
    grid        = budget_grid_df, 
    media_spend =  media_spend_df,
    verbose     = True
)              
    
budget_optimized['media_spend_rebal']\
    .melt(id_vars = 'media')\
        .pipe(px.bar,
              x = 'variable',
              y = 'value',
              color = 'media',
              barmode = 'group',
              template = 'plotly_dark')

# Save work

from joblib import dump, load

dump(best_model, "artefacts/best_model")

load("artefacts/best_model")

dump(media_spend_df,"artefacts/media_spend_df")
load("artefacts/media_spend_df")

dump(x_train,"artefacts/X_train")

load("artefacts/X_train")


