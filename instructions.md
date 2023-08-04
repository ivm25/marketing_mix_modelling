## Project Setup:
- If you dont have python installed, please head over to anaconda and install it first. Use this link for relevant python installers:
  
  https://www.anaconda.com/download
   
- You should lock down your environment with a `requirements.txt` file. The requrements file is present within the repository

### Activate virtual env:

```
pip install -r /path/to/requirements.txt
```

Or

Create an environment.yml file similar to the one created in this repository. Follow the instructions on top of the .yml file to run the relevant commands and create an environment for this project.

# VS Code 

If you are using VS Code as your preferred IDE, make sure to add/edit your environment variables and then run the scripts.

- Use this link to overcome the issue of conda not found error and edit your environment variables accordingly: https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10

## Input Data (How to use it and what to do in case of difference)

- The current data file is present in the folder (data) and named as "de_simulated_data"
- Data consists of the following columns:
  - Revenue (Key Metric)
    - Date (Weekly dates in the format yyyy-mm-dd)
    - TV_s (Spends on TV marketing campaigns)
    - OOH_s (Spends on OOH marketing campaigns)
    - Print_s (Spends on Print marketing campaigns)
    - Search_s (Spends on Search marketing campaigns)
    - Facebbok_s (Spends on Facebook marketing campaigns)
    - Competitor_sales_b (Sales from different competitor products)
    - Search_clicks (Clicks on Search marketing campaigns)
    - Facebbok_I (Number of Facebook Impressions)

- In case of different data columns (For example: Instagram_spends or Instagram_clicks):
    - Place where to modify the code:
    - In (**app.py, main.py, data_prep.py, models.py**,  remove "Instgram_I" like columns by adding to the list in the columns to 
      drop list (when correlation_df is created)
    - Apart from the above point, the intake of rest of the columns should not be affected and the models should be able to process data 
      regardless of the difference.

  


## Building the app with `Dash`
    - I have used render.com to host the dash app. It is a relatively simpler way to host any plotly dash web apps. Use this video link to understand further: https://www.youtube.com/watch?v=XWJBJoV5yww
    - It invloves creating a similar requirements.txt file and seeting up a free website app hosting session on render
    - Alternatively, you could use Azure/GCP or AWS free compute engines to host your app.
  



