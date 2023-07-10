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
from data_prep.data_prep import generate_grid, adstock_model, extract_coefficients

# Data load

simulated_data_df = pd.read_csv("data/de_simulated_data.csv")

simulated_data_df.columns = simulated_data_df.columns.str.lower()
simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])   
 
correlation_df = simulated_data_df\
    .assign(date_month = lambda x:x['date'].dt.month_name())\
        .drop(["date","search_clicks_p","facebook_i"], axis = 1)



# APP SETUP
external_stylesheets = [dbc.themes.ZEPHYR]

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

# importing models from the prep work

grid = generate_grid(size = 100)

ols_models = adstock_model(correlation_df[0:100],
                           grid)

key_coefficients_dict = extract_coefficients(model_dict = ols_models)

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


cons = sm.add_constant(x_train)

model_ols = sm.OLS(y_train, x_train).fit()


# print(model_ols.summary())

output_summary_stats = model_summary_to_dataframe(model_ols)    

font_colour = "RebeccaPurple"

table = go.Figure(data=[go.Table(
    header=dict(values=list(model_ols.summary().tables[0].data[0]),
                fill = dict(color = font_colour),
                font = dict(color = 'rgb(255,255,255)')),
    cells=dict(values=list(zip(*model_ols.summary().tables[0].data[1:])),
               fill = dict(color='rgb(245,245,245)'),
               font= dict(family="Courier New, monospace", size=14, color='rgb(0,0,0)'))
)])

table.update_layout(template = 'ggplot2',
                    title = "Multiple Linear Regression output for adstock <b>0.50</b><br><sup>R square value of <b>0.910</b></sup>",
                    font=dict(
                    family="Courier New, monospace",
                    size=14,
                    ),
                    title_font_color = "RebeccaPurple"
                    )
      
table_2 = go.Figure(data=[go.Table(
    header=dict(values=list(model_ols.summary().tables[1].data[0]),
                fill = dict(color = font_colour),
                font = dict(color = 'rgb(255,255,255)')),
    cells=dict(values=list(zip(*model_ols.summary().tables[1].data[1:])),
               fill = dict(color='rgb(245,245,245)'),
               font= dict(family="Courier New, monospace", size=14, color='rgb(0,0,0)'))
)])

table_2.update_layout(template = 'ggplot2',
                    title = "Signifiance values for adstock <b>0.50</b><br><sup>R square value of <b>0.910</b></sup>",
                    font=dict(
                    family="Courier New, monospace",
                    size=14,
                    ),
                    title_font_color = "RebeccaPurple"
                    )


# data prep to show coefficients
# key_coefficients = pd.DataFrame(model_ols.summary().tables[1])\
#                     .iloc[1:6,0:2]\
#                     .rename(columns={0:'variable', 1:'coefficient'})
                    
                    
# key_coefficients['coefficient'] = key_coefficients['coefficient'].astype(str)
# key_coefficients['coefficient'] = key_coefficients['coefficient'].astype(float)
# key_coefficients['variable'] = key_coefficients['variable'].astype(str)
# key_coefficients['key variable'] = key_coefficients['variable'].str[8:]

# figure_2 = px.bar(key_coefficients,
#                   x = 'key variable',
#                   y ='coefficient',
#                   color = 'coefficient',
#                   text = 'coefficient'
#                   )
# figure_2.update_layout(
#     title="Coefficeint Values for adstock = <b>0.50</b>",
#     # xaxis_title="key variables",
#     yaxis_title="coefficients",
#     template = 'ggplot2',
#     legend_title="Legend",
#     font=dict(
#         family="Courier New, monospace",
#         size=16,
#         color="RebeccaPurple"
#     )
# )

# figure_2.update_xaxes(showticklabels=True)
# figure_2.update_layout(showlegend=True)



app.layout = html.Div(children=[navbar,
    html.H4("Understanding the impact of different marketing mix strategies on sales"),
    dbc.Row([dbc.Col(html.P("Select a variable:"))]),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='dropdown',
        options=list(correlation_df.columns),
        value=correlation_df.columns[4],
         style={"width": "600px"},
        
       
       
    )),dbc.Col(dcc.Dropdown(
        id='dropdown_adstck',
        options=list(key_coefficients_dict.keys()),
        value=0.1791267574878015,
         style={"width": "600px"},
        
       
       
    ))]),
    dbc.Row([dbc.Col(dcc.Graph(id="graph", style = {'display': 'inline-block'})),
    dbc.Col(dcc.Graph(id = "graph_2", style ={'display': 'inline-block'}))]),
    dbc.Row([dbc.Col(dcc.Graph(figure = table)),
             dbc.Col(dcc.Graph(figure = table_2))])
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
    title="Revenue as a function of media spends <br><sup><b>Search Media</b> has the highest R square value</sup>",
    xaxis_title="Variable spends",
    yaxis_title="Revenue", 
    legend_title="Legend",
    template = 'ggplot2',
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="RebeccaPurple"
    )  
)
    fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',.')
    fig.update_layout(xaxis_tickprefix = '$', xaxis_tickformat = ',.')
    return fig


@app.callback(
    Output(component_id="graph_2", component_property="figure"),
    Input(component_id="dropdown_adstck", component_property="value")
)

def scenario_adstock(selected_key):
  
    if selected_key is None:
        selected_key = 0.1791267574878015
    selected_data = pd.DataFrame(key_coefficients_dict.get(selected_key))


    figure_2 = px.bar(selected_data,
                    x = selected_data["key variable"],
                    y = selected_data["coefficient"],
                    color = 'coefficient',
                    text = 'coefficient'
                    )
    figure_2.update_layout(
        title="Coefficeint Values for adstock = " + selected_key,
        # xaxis_title="key variables",
        yaxis_title="coefficients",
        template = 'ggplot2',
        legend_title="Legend",
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="RebeccaPurple"
        )
    )

    figure_2.update_xaxes(showticklabels=True)
    figure_2.update_layout(showlegend=True)
    return figure_2


if __name__ == '__main__':
    app.run_server(debug = True)

    