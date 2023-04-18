#Import libraries

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
from dash import Dash

import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
import pathlib
from joblib import load
import json

# Data load

simulated_data_df = pd.read_csv("data/de_simulated_data.csv")
simulated_data_df['date'] = pd.to_datetime(simulated_data_df['DATE'])    
    
    
# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__,
           external_stylesheets=external_stylesheets)

PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'

# PATHS

BASE_PATH =pathlib.Path(__file__).parent.resolve()

ART_PATH = BASE_PATH.joinpath("artefacts").resolve()

# ARTEFACTS

best_model = load(ART_PATH.joinpath("best_model"))

X_Train = load(ART_PATH.joinpath("X_train"))

media_spend_df = load(ART_PATH.joinpath("media_spend_df"))

# FUNCTIONS (make them modular later on)

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

# APP
navbar = dbc.Navbar(
         [html.A(dbc.Row([dbc.Row(dbc.NavbarBrand("Marketing Mix Optimisation", className= "ml-2"))
                          ],
                         align = "center",
                         no_gutters=True)
                 ),
          dbc.NavbarToggler(id="navbar-toggler", n_clicks = 0),
          dbc.Collapse(
              id = "navbar-collapse", navbar = True, is_open = False
          )
          ], 
         color = "dark",
         dark = True
          )


app.layout = html.Div(children=[navbar,dbc.Row(
                                               [
                                                dbc.Col(
                                                [html.H3(children = 'Welcome to MMM Optimisation dashboard'),
                                                html.Div(
                                                    id = "intro",
                                                    children = "Improve Estimated Revenue by Optimising media channel spend and budget.",
                                                ),
                                                html.Br(),
                                                html.Hr(),
                                                html.H5("Generate Random Media Spend Portfolios"),
                                                html.P("Increase the number of media portfolios generated to improve the optimisation results"),
                                                dcc.Slider(
                                                    id = "budget-size",
                                                    min = 1,
                                                    max = 5000,
                                                    step = 10,
                                                    marks = {
                                                        1:"1",
                                                        2000:"2000",
                                                        3000: "3000",
                                                        4000: "4000",
                                                        5000: "5000"
                                                    }, 
                                                    value =10
                                                ),
                                                html.Hr(),
                                                html.H5("Score"),
                                                html.P("A score of 1.02 indicates a predicted 2% increase in Revenue by optimising the marketing budget"),
                                                daq.LEDDisplay(
                                                    id = "digital-score",
                                                    value = 1.09,
                                                    color = "#92e0d3",
                                                    backgroundColor = "#1e2130",
                                                    size =50
                                                ),
                                                html.Br(),
                                                html.Button("Download Media Spend", id = "btn"),
                                                dcc.Download(id = "download")
                                                
                                                ],
                                                width = 3,
                                                style = {'margin': '10px'}
                                                ),
                                                dbc.Col(
                                                    dcc.Graph(id ='spend-graph'),
                                                    width =3
                                                ),
                                                dbc.Col(dcc.Graph(id = 'time_series'),
                                                        width = 5),
                                                
                                                
                                                dcc.Store(id = 'intermediate-value'),
                                                dcc.Store(
                                                          
                                                           id = 'time_series_slider'
                                                           )
                                              
                                                ]
                                               )
                                ]
                      )


# Writing Callbacks
@app.callback(
   Output(component_id='intermediate-value',component_property='data'),
   Input(component_id='budget-size', component_property='value')
)


def process_data(budget_size):
    budget_grid_df = generate_grid(budget_size)
    
    budget_optimised = optimize_budget(
        df = X_Train,
        grid = budget_grid_df,
        media_spend = media_spend_df,
        verbose = True 
    )
    
    budget_optimised_json = budget_optimised.copy()
    
    budget_optimised_json['rebalancing_coef'] = budget_optimised_json['rebalancing_coef'][0].to_json()
    
    budget_optimised_json['media_spend_rebal'] = budget_optimised_json['media_spend_rebal'].to_json()
    
    budget_optimised_json = json.dumps(budget_optimised_json)
    
    return budget_optimised_json
    

@app.callback(
    Output(component_id='spend-graph', component_property='figure'),
    Input(component_id='intermediate-value', component_property='data')
)



def update_figure(budget_optimised_json):
    
    budget_optimised = json.loads(budget_optimised_json)
    
    fig =pd.read_json(budget_optimised['media_spend_rebal'])\
        .melt(
            id_vars='media'
        )\
            .pipe(
                px.bar,
                x ='variable',
                y = 'value',
                color = 'media',
                barmode = 'group',
                template = 'plotly_dark'
            )
    return fig



@app.callback(
    Output(component_id='time_series', component_property='figure'),
    Input(component_id='time_series_slider', component_property='data')
)



def update_correlation_figure(df):
    
    
    df = simulated_data_df
    
    fig2 = df\
            .melt(id_vars = "date")\
                .pipe(px.line,
                    x = "date",
                    y = "value",
                    color = "variable",
                    facet_col = "variable",
                    facet_col_wrap = 3,
                    template = "plotly_dark")\
                        .update_yaxes(matches = None)



    return fig2





# @app.callback(
#     Output(component_id='digital-score', component_property='value'),
#     Input(component_id='intermediate-value', component_property='data')
# )

# def update_digital(budget_optimised_json):
    
#     budget_optimised = json.loads(budget_optimised_json)
    
#     return np.round(float(budget_optimised_json['score']),2)


# Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug = False)
        
    