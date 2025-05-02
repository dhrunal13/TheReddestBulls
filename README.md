# The Reddest Bulls 
**Forecasting FX Rates Using Macroeconomic Fundamentals**

The project analyzes how U.S. macroeconomic indicators influence major foreign exchange (FX) rates relative to the U.S. Dollar.  
It combines data science, economic theory, and scenario simulation into an interactive dashboard that forecast rates based on the trained models (OLS, Lasso, LassoCV, XGboost).

---

## Dashboard Link

Access the deployed dashboard here:  
**[The Reddest Bulls Dashboard](https://thereddestbulls.streamlit.app/)**  

---

## File Structure

| File/Folder                     | Description |
|--------------------------------|-------------|
| `dashboard.py`                 | Main Streamlit app used to launch the dashboard |
| `run_forex_model.py`           | ML engine that trains models, evaluates predictions, and forecasts FX rates |
| `requirements.txt`             | Python dependencies for local or cloud deployment |
| `macro_data.csv`               | Cleaned macroeconomic dataset with all features used in modeling |
| `forex_merged_cleaned.csv`     | Cleaned FX log returns for all USD currency pairs |
| `raw_data/`                    | Contains all raw macroeconomic and FX data, plus original data descriptors |
| `Logo.png`                     | The COOLEST project logo (courtesy of OpenAi) used in the dashboard header |
| `Untitled.png`                 | Static heatmap image used in the Static Analysis tab |
| `macro_eda_report.html`        | Full EDA report for macroeconomic variables |
| `forex_eda_report.html`        | Full EDA report for FX returns |
| `EDA-Macroeconomic-Factors.ipynb` | Notebook for macro data cleaning and EDA |
| `EDAForex.ipynb`               | Notebook for FX data cleaning and EDA |
| `MLmodel.ipynb`                | Initial experiments and visualizations for feature creation |
| `MLModelMaybe.ipynb`           | Final refined modeling prototype (OLS, Lasso, LassoCV, XGboost) |
| `proposal.md`                  | Project proposal describing motivation and direction |

---

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/dhrunal13/thereddestbulls.git
   cd thereddestbulls
   ```
   
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the dashboard:

    ```bash
    streamlit run dashboard.py
    ```
    
## What the Dashboard Does

- Visualizes FX return behavior across currency pairs
- Trains OLS, Lasso, LassoCV and XGboost models on macroeconomic predictors
- Allows scenario-based forecasting over a 1-20 years (user can pick)
- Lets users interactively adjust macro conditions and explore projected currency movements
- Includes detailed model performance, correlation heatmaps, and downloadable reports
