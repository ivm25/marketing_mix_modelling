# IMPORT ALL THE LIBRARIES


import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error





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
    return media_figure


# Correlations ----

def corelation_plot(data):
    corelation_plot = data\
        .corr()\
            .pipe(sns.heatmap,
                annot =True)
    return corelation_plot


