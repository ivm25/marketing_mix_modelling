#Import libraries

import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import Dash

import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
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

from models.models import model_summary_to_dataframe
from models.models import adstock
from visualisations.charts import time_series_chart, visualise_media_spend, corelation_plot


# Data load

simulated_data_df = pd.read_csv("data/de_simulated_data.csv")

simulated_data_df.columns = simulated_data_df.columns.str.lower()
simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])   
 
correlation_df = simulated_data_df\
    .assign(date_month = lambda x:x['date'].dt.month_name())\
        .drop(["date","search_clicks_p","facebook_i"], axis = 1)



# APP SETUP
external_stylesheets = [dbc.themes.JOURNAL]

app = Dash(__name__,
           external_stylesheets=external_stylesheets)

server = app.server


# APP
navbar = dbc.Navbar(
         [html.A(dbc.Row([dbc.Row(dbc.NavbarBrand("Marketing Mix Modelling", className= "ml-2"))
                          ],
                         align = "center"
                         )
                 ),
          dbc.NavbarToggler(id="navbar-toggler", n_clicks = 0),
          dbc.Collapse(
              id = "navbar-collapse", navbar = True, is_open = False
          )
          ], 
        #  color = "dark",
        #  dark = True
          )



# Create a table with the model summary
df = correlation_df
adstock_tv_s = adstock(df.tv_s, 0.5)
adstock_ooh_s = adstock(df.ooh_s, 0.5)

adstock_print_s = adstock(df.print_s, 0.5)
adstock_search_s = adstock(df.search_s, 0.5)

adstock_facebook_s = adstock(df.facebook_s, 0.5)



# prep fpr modelling

X = pd.concat([adstock_tv_s, 
            adstock_ooh_s, 
            adstock_print_s,
            adstock_search_s,
            adstock_facebook_s,
            df.competitor_sales_b,
            pd.get_dummies(df.date_month)
            ], axis = 1)

Y = df.revenue
x_train, x_test, y_train, y_test = train_test_split(X,
                                                Y,
                                                random_state = 42)

melted_training_data = pd.melt(x_train.reset_index(),
                                id_vars = 'index')

cons = sm.add_constant(x_train)

model_ols = sm.OLS(y_train, x_train).fit()


# print(model_ols.summary())

output_summary_stats = model_summary_to_dataframe(model_ols)    


table = go.Figure(data=[go.Table(
    header=dict(values=list(model_ols.summary().tables[0].data[0])),
    cells=dict(values=list(zip(*model_ols.summary().tables[0].data[1:])))
)])

table.update_layout(template = 'ggplot2')


key_coefficients = pd.DataFrame(model_ols.summary().tables[1])\
                    .iloc[1:7,0:2]\
                    .rename(columns={0:'variable', 1:'coefficient'})
                    
                    
key_coefficients['coefficient'] = key_coefficients['coefficient'].astype(str)
key_coefficients['coefficient'] = key_coefficients['coefficient'].astype(float)
key_coefficients['variable'] = key_coefficients['variable'].astype(str)

figure_2 = px.bar(key_coefficients,
                  x = 'variable',
                  y ='coefficient',
                  color = 'coefficient'
                  )
figure_2.update_layout(
    title="Coefficeint Values",
    xaxis_title="key variables",
    yaxis_title="coefficients",
    template = 'ggplot2',
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"
    )
)

independent_vars = {'TV': correlation_df['tv_s'],
                    'OOH':correlation_df['ooh_s']}


app.layout = html.Div(children=[navbar,
    html.H4("Understanding the impact of different marketing mix strategies on sales"),
    dbc.Row([dbc.Col(html.P("Select the variable:"))]),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='dropdown',
        options=list(correlation_df.columns),
        value=correlation_df.columns[1]
       
    ))]),
    dbc.Row([dbc.Col(dcc.Graph(id="graph", style = {'display': 'inline-block'})),
    dbc.Col(dcc.Graph(figure = figure_2, style ={'display': 'inline-block'}))]),
    dcc.Graph(figure = table)
])

# Writing Callbacks

@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="dropdown", component_property="value")
)

def mod(selected_col):
    # scatter charts with linear fits
    if selected_col is None:
        selected_col = correlation_df.columns[1]
   
    
    fig = px.scatter(correlation_df,
                     x = correlation_df[selected_col],
                     y = 'revenue',
                     trendline = 'ols',
                     size = 'revenue',
                     template = 'ggplot2',
                     color = 'revenue')
    

    fig.update_layout(
    title="Key performance indicators",
    xaxis_title="Variable spends",
    yaxis_title="sales",
    legend_title="Legend",
    template = 'ggplot2',
    font=dict(
        family="Courier New, monospace",
        size=20,
        color="RebeccaPurple"
    )
)
     
    return fig



if __name__ == '__main__':
    app.run_server(debug = True)

    