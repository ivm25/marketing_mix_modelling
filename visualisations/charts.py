# IMPORT ALL THE LIBRARIES


import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from data_prep.data_prep import time_series_prep



def time_series_chart(data):

# Visualize Time Series Data ----
    time_series_figure = data\
        .melt(id_vars = "date")\
            .pipe(px.line,
                x = "date",
                y = "value",
                color = "variable",
                facet_col = "variable",
                facet_col_wrap = 3,
                template = "plotly_dark")\
                    .update_yaxes(matches = None)
    return time_series_figure



# Visualize Current Spend Profile ----

def visualise_media_spend(media_data):
    media_figure = media_data\
        .pipe(px.bar,
            x = 'media',
            y = 'spend_current',
            color = 'media',
            template = "plotly_dark"
            )
    media_figure.show()
    return media_figure


# Correlations ----

def corelation_plot(data):
    corelation_plot = data\
        .corr()\
            .pipe(sns.heatmap,
                annot =True)
    return corelation_plot


def marginal_plots(data):
    figure = px.density_heatmap(data, x="tv_s",
                                y="revenue", 
                                marginal_x="box",
                                marginal_y="violin")
    figure.show()

def heatmaps(df):
    heatmap = px.imshow(df, text_auto=True)
    heatmap.show()
    return heatmap

def seasonality_charts(time_series_data):
    
    return (make_subplots(
            rows = 4,
            cols = 1,
            subplot_titles=["Observed",
                            "Trend",
                            "Seasonal",
                            "Residuals"]
           )
           .add_trace(
               go.Scatter(x = time_series_data.observed.index,
                          y = time_series_data.observed),
               row = 1,
               col = 1,
           ).add_trace(
               go.Scatter(x = time_series_data.trend.index,
                          y = time_series_data.trend),
               row = 2,
               col = 1,
           ).add_trace(
               go.Scatter(x = time_series_data.seasonal.index,
                          y = time_series_data.seasonal),
               row = 3,
               col = 1,
           ).add_trace(
               go.Scatter(x = time_series_data.resid.index,
                          y = time_series_data.resid),
               row = 4,
               col = 1,
           )
    )
    