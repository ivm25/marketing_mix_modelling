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
from plotly.subplots import make_subplots

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
from data_prep.data_prep import generate_grid, adstock_model, extract_coefficients, time_series_prep

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

time_series_dict = time_series_prep(simulated_data_df)

# Create a table with the model summary

df = correlation_df
font_colour = "RebeccaPurple"





app.layout = html.Div(children=[navbar,
    html.H4("Understanding the impact of different marketing mix strategies on sales"),
    dbc.Row([dbc.Col(html.P("Select a variable:")), dbc.Col(html.P("Select an adstock scenario:"))]),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='dropdown',
        options=list(correlation_df.columns[0:6]),
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
    dbc.Row([dbc.Col(dcc.Dropdown(
        id='dropdown_heatmap',
        options=list(correlation_df.columns[0:6]),
        value=correlation_df.columns[4],
         style={"width": "600px"},
        
       
       
    )),dbc.Col(dcc.Dropdown(
        id='time_series_dropdown',
        options=list(time_series_dict.keys()),
        value=4,
         style={"width": "600px"},
        
       
       
    ))]),
    dbc.Row([dbc.Col(dcc.Graph(id = "heatmap", style = {'display': 'inline-block'})),
             dbc.Col(dcc.Graph(id = "time_series", style = {'display': 'inline-block'}))])
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
    xaxis_title= selected_col + "pends",
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
        title="Coefficeint Values for the selected adstock<br><sup><b>TV and Facebook</b> have the highest ROI for every $ spent</sup>",
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




@app.callback(
    Output(component_id="heatmap", component_property="figure"),
    Input(component_id="dropdown_heatmap", component_property="value")
)

def heatmap_table(selected_col):
    
   if selected_col is None:
        selected_col = correlation_df.columns[1]

   heatmap = px.density_heatmap(correlation_df,
                                 x = correlation_df[selected_col],
                                 y = 'revenue',
                                 marginal_x= 'violin',
                                 marginal_y= 'violin'
                                 
    )

   heatmap.update_layout(template = 'ggplot2',
                        title = "Distribution of points",
                        font=dict(
                        family="Courier New, monospace",
                        size=14,
                        color="RebeccaPurple"
                        ),
                        title_font_color = "RebeccaPurple"
                        )
   return heatmap

@app.callback(
    Output(component_id="time_series", component_property="figure"),
    Input(component_id="time_series_dropdown", component_property="value")
)


def seasonality_charts(time_series_data_key):
    if time_series_data_key is None:
        time_series_data_key = 4
        
    decomposed_data = time_series_dict.get(time_series_data_key)
    
    time_series_fig = make_subplots(
    rows = 4,
    cols = 1,
    subplot_titles=["Total Revenue",
                    "Overall Trend",
                    "Seasonality",
                    "Revenue due to Marketing Channels"],
    vertical_spacing=0.20
    ).add_trace(
        go.Scatter(x = decomposed_data.observed.index,
                    y = decomposed_data.observed),
        row = 1,
        col = 1,
    ).add_trace(
        go.Scatter(x = decomposed_data.trend.index,
                    y = decomposed_data.trend),
        row = 2,
        col = 1,
    ).add_trace(
        go.Scatter(x = decomposed_data.seasonal.index,
                    y = decomposed_data.seasonal),
        row = 3,
        col = 1,
    ).add_trace(
        go.Scatter(x = decomposed_data.resid.index,
                    y = decomposed_data.resid,
                     ),
        row = 4,
        col = 1,
    )
    
    time_series_fig.update_layout(template = 'ggplot2',
                        title = "Impact of Marketing and Non Marketing Media on Revenue",
                        font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="RebeccaPurple"
                        ),
                        title_font_color = "RebeccaPurple",
                        showlegend = False
                        
                        )

    
    
    return time_series_fig
    



# @app.callback(
#     Output(component_id="table_1", component_property="figure"),
#     Input(component_id="dropdown_adstck", component_property="value")
# )

# def rsquared_table(selected_key):
    
#     if selected_key is None:
#         selected_key = 0.1791267574878015
#     selected_model = ols_models.get(selected_key)

#     table = go.Figure(data=[go.Table(
#             header=dict(values=list(selected_model.summary().tables[0].data[0]),
#                     fill = dict(color = font_colour),
#                     font = dict(color = 'rgb(255,255,255)')),
#             cells=dict(values=list(zip(*selected_model.summary().tables[0].data[1:])),
#                 fill = dict(color='rgb(245,245,245)'),
#                 font= dict(family="Courier New, monospace", size=14, color='rgb(0,0,0)'))
#     )])

#     table.update_layout(template = 'ggplot2',
#                         title = "Multiple Linear Regression output for adstock =" + str(round(selected_key,2)),
#                         font=dict(
#                         family="Courier New, monospace",
#                         size=14,
#                         ),
#                         title_font_color = "RebeccaPurple"
#                         )
#     return table






# @app.callback(
#     Output(component_id="table_2", component_property="figure"),
#     Input(component_id="dropdown_adstck", component_property="value")
# )

# def significance_values(selected_key):
    
#     if selected_key is None:
#         selected_key = 0.1791267574878015
#     selected_model = ols_models.get(selected_key)
    
#     table_2 = go.Figure(data=[go.Table(
#         header=dict(values=list(selected_model.summary().tables[1].data[0]),
#                     fill = dict(color = font_colour),
#                     font = dict(color = 'rgb(255,255,255)')),
#         cells=dict(values=list(zip(*selected_model.summary().tables[1].data[1:])),
#                 fill = dict(color='rgb(245,245,245)'),
#                 font= dict(family="Courier New, monospace", size=14, color='rgb(0,0,0)'))
#     )])

#     table_2.update_layout(template = 'ggplot2',
#                         title = "Signifiance values for adstock =" + str(round(selected_key,2)),
#                         font=dict(
#                         family="Courier New, monospace",
#                         size=14,
#                         ),
#                         title_font_color = "RebeccaPurple"
#                         )
#     return table_2


if __name__ == '__main__':
    app.run_server(debug = True)

    