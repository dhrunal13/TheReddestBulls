# dashboard.py

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from pathlib import Path  # Fix: import Path here
# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="The Reddest Bulls", layout="wide")

# ---------------------------------
# Optional: Set Background (still looks good)
# ---------------------------------
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1565372870225-2bbae36c9f4f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            opacity: 0.95;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ---------------------------------
# Main Title (Shown on every page)
# ---------------------------------
logo = Image.open("Logo.png")  # Your logo file

st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://raw.githubusercontent.com/dhrunal13/thereddestbulls/main/Logo.png" alt="Logo" style="height:150px;">
        <div>
            <h1 style="margin: 0; color: #C41E3A;">The Reddest Bulls</h1>
            <h4 style="margin: 0; color: gray;">Macroeconomic Drivers of FX Rates</h4>
        </div>
    </div>
    <hr style='border:1px solid #C41E3A'>
""", unsafe_allow_html=True)
# ---------------------------------
# Top Navigation Tabs
# ---------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "Hypotheses", 
    "Data Explorer", 
    "Static Analysis", 
    "Forecasting Models", 
    "Scenario Simulator"
])

# ---------------------------------
# Page Content
# ---------------------------------
with tab1:
    st.markdown("## Project Introduction")
    st.write("""
    This dashboard analyzes how U.S. macroeconomic fundamentals impact foreign exchange (FX) rates relative to the U.S. Dollar.
    
    We apply economic theory, statistical modeling, and machine learning techniques to:
    
    - Analyze key macro indicators such as interest rates, inflation, and industrial production
    - Predict currency movements using OLS regression and advanced forecasting models
    - Simulate macroeconomic scenarios to forecast FX reactions
    """)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("## Why this Project Matters")
    st.write("""
    Understanding these dynamics is critical for:
    
    - Investors evaluating currency exposure and international investments
    - Policy makers designing monetary and fiscal interventions
    - Global businesses managing FX risk and operational exposure
    """)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("## Dashboard Navigation Guide")
    st.write("""
    - **Hypotheses**: Explore the economic theories and relationships we are testing.
    - **Data Explorer**: Visualize historical FX rates and macroeconomic indicators.
    - **Static Analysis**: Review OLS regression outputs and significance testing.
    - **Forecasting Models**: Compare the predictive accuracy of OLS versus machine learning models.
    - **Scenario Simulator**: Adjust macro variables and simulate FX outcomes dynamically.
    
    ---
    
    **Note:** Use the tabs above to navigate between sections.
    """)

with tab2:
    st.markdown("## Hypotheses")
    st.write("""
    This section outlines the key hypotheses guiding our analysis:

    1. **Interest Rate Hypothesis**: Higher U.S. interest rates strengthen the USD.
    2. **Inflation Hypothesis**: Rising inflation weakens the USD relative to other currencies.
    3. **Industrial Production Hypothesis**: Higher U.S. output correlates with a stronger USD.
    4. **Risk Aversion Hypothesis**: During global uncertainty, the USD behaves as a safe-haven.
    
    Future design:
    - Shortcut buttons will link to relevant analyses and forecasts.
    """)

with tab3:
    st.markdown("## Data Overview and Explorer")

    st.write("""
    This section provides an overview of the raw datasets and gives you access to download the full EDA reports.
    
    **Data Sources:**
    - Macroeconomic Factors: Sourced from FRED. Includes indicators like Interest Rate, Inflation, Industrial Production, and others.
    - Forex Rates: Historical currency exchange rates for major pairs such as USD-EUR, USD-JPY, etc.

    Both datasets were cleaned, merged, and used for modeling and forecasting.
    """)

    st.markdown("---")
    st.markdown("### Macroeconomic Factors Description")
    st.markdown("""
    <div style='
        padding: 15px; 
        background-color: #f9f9f9; 
        border-left: 4px solid #C41E3A;
        border-radius: 4px;
        font-size: 16px;
        line-height: 1.6;
    '>
        The macroeconomic dataset contains monthly data from FRED on key indicators such as interest rates, inflation, 
        industrial production, trade balance, unemployment, consumer sentiment, and risk indices like the VIX and S&P 500. 
        These variables were selected due to their theoretical and empirical linkages with FX movements and were preprocessed 
        to remove missing values and align frequencies for modeling purposes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Forex Rates Description")
    st.markdown("""
    <div style='
        padding: 15px; 
        background-color: #f9f9f9; 
        border-left: 4px solid #C41E3A;
        border-radius: 4px;
        font-size: 16px;
        line-height: 1.6;'>
    The forex dataset includes monthly exchange rates for ten major USD pairs (e.g., USD-EUR, USD-JPY, USD-XAU). 
    Each pair was transformed into log returns to stabilize variance and capture relative changes in value. 
    These returns were then aligned with the macroeconomic data for use in forecasting models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Raw Data Snapshots")

    st.markdown("#### Macro Data Preview")
    try:
        macro_df = pd.read_csv("macro_data.csv", parse_dates=["DATE"])
        st.dataframe(macro_df.head())
    except FileNotFoundError:
        st.error("Macro data CSV not found.")

    st.markdown("#### Forex Data Preview")
    try:
        forex_df = pd.read_csv("forex_merged_cleaned.csv", parse_dates=["DATE"])
        st.dataframe(forex_df.head())
    except FileNotFoundError:
        st.error("Forex data CSV not found.")

    st.markdown("### Download Full EDA Reports")

    try:
        with open("macro_eda_report.html", "rb") as macro_file:
            st.download_button("Download Macro EDA Report", macro_file, file_name="macro_eda_report.html", mime="text/html")
    except FileNotFoundError:
        st.error("Macro EDA HTML file not found.")

    try:
        with open("forex_eda_report.html", "rb") as forex_file:
            st.download_button("Download Forex EDA Report", forex_file, file_name="forex_eda_report.html", mime="text/html")
    except FileNotFoundError:
        st.error("Forex EDA HTML file not found.")

with tab4:
    st.markdown("## Static Analysis")

    st.write("""
    Review results from OLS regressions measuring the sensitivity of FX rates to macroeconomic variables.
    This section presents the results of our OLS regressions, highlighting how key U.S. macroeconomic factors — 
    such as interest rates, inflation, industrial production, and risk sentiment — influence major USD currency pairs.

    The heatmap below displays the regression coefficients across a 10x10 macro-currency grid.
    Values are underlined if they are statistically significant at the 5% level, helping to pinpoint which relationships are 
    both economically meaningful and statistically robust.
    """)

    # Coming features list
    st.markdown("""
    **Features coming soon:**
    - Coefficient tables
    - P-values and significance highlights
    - Cross-currency comparisons
    """)

    # Coefficient heatmap
    st.image("Untitled.png", caption="OLS Coefficient Heatmap (Underline = Stat. Significant)", use_column_width=True)

    # Interpretation guide
    st.write("""
    **How to read this heatmap:**
    - Each cell shows the estimated coefficient from the OLS regression for a given macro factor on a specific currency pair.
    - Red shades indicate a positive relationship, blue shades indicate a negative relationship.
    - Underlined values mark coefficients that are statistically significant (p < 0.05).
    
    This static analysis helps uncover cross-currency patterns and the relative importance of different macroeconomic drivers.
    """)


with tab5:
    st.markdown("## How the Model Works")

    st.write( """
    This section provides a complete explanation of how we built, trained, and used our models to analyze and forecast foreign exchange (FX) rates.

    We designed the modeling process to be flexible, interpretable, and aligned with economic intuition. Here's how it works from start to finish:

    ### 1. Data Collection and Integration

    We use two main sources of data:

    - **Macroeconomic Indicators:** These include variables like interest rates, inflation (CPI), industrial production, unemployment, consumer sentiment, and more. They are pulled from public datasets, mainly the Federal Reserve Economic Data (FRED).
    
    - **Foreign Exchange Rates:** Monthly FX rates for 10 major currency pairs against the U.S. Dollar (e.g., USD-EUR, USD-JPY, USD-GBP). We cleaned and standardized them for analysis.

    The two datasets are merged by date to create one unified dataset that captures both economic drivers and market outcomes.

    ### 2. Why We Use Log Returns

    Exchange rates are typically non-stationary, meaning their average and variance change over time. This makes prediction difficult. To address this, we convert exchange rate values into **log returns**, which are more stable and interpretable.

    Log returns help us focus on relative changes, like “how much did the currency move this month compared to last month,” rather than absolute price levels.

    ### 3. Feature Engineering

    Economic changes often don’t affect currency markets instantly. For example, a rise in interest rates may take several months to influence exchange rates.

    To capture this delayed effect, we add:

    - **Lagged values** (1 to 60 months): These let the model look back in time to see recent trends.
    - **Rolling averages and standard deviations:** These summarize the recent behavior of macro indicators — how fast they’re rising, how volatile they are, etc.

    These added features help the model understand patterns over time, not just one snapshot.

    ### 4. Train/Test Split

    To evaluate how well our models generalize to new data, we split the dataset:

    - **Training data:** Includes all observations up to December 2022
    - **Testing data:** Covers January 2023 onward

    We train the models only on the training set, and then test them on unseen data to evaluate performance fairly.

    ### 5. Model Options and Why We Chose Them

    The dashboard supports three types of models. Each serves a different purpose:

    - **Ordinary Least Squares (OLS):**
        - Simple linear regression
        - Easy to interpret
        - Good for understanding the strength and direction of relationships
        - However, it may struggle when too many features are added or when the relationships are non-linear

    - **Lasso Regression:**
        - A version of linear regression that penalizes complexity
        - Helps automatically select the most important features
        - Useful when you have many input variables (which we do after feature engineering)

    - **XGBoost:**
        - A powerful machine learning model that builds decision trees
        - Very effective for capturing complex, non-linear relationships
        - Often gives the best accuracy but is harder to interpret

    Users can switch between these models to compare performance and see which performs better for a given currency and set of macro indicators.

    ### 6. Model Evaluation

    After training, we evaluate the models using the testing data. We look at:

    - **R² Score:** How well the model explains the variation in returns (higher is better)
    - **Mean Absolute Error (MAE):** Average size of prediction errors
    - **Root Mean Squared Error (RMSE):** Similar to MAE but penalizes large errors more

    These metrics are displayed in the dashboard so users can understand how reliable each model is.

    We also show a line graph comparing **actual vs predicted** returns on the test set, so users can visually assess how close the model gets to reality.

    ### 7. Forecasting Future Exchange Rates

    Once a model is trained, we use it to simulate future FX rates. This is how:

    - We start with the most recent known macroeconomic values (as of December 2022)
    - Users can adjust those values using sliders (e.g., increase interest rate by 0.5%)
    - We apply the model to forecast monthly returns for the next 1 to 20 years
    - These returns are converted back into forecasted exchange rates using compounding

    The dashboard plots these forecasts and allows users to download the data.

    ### 8. Summary

    In simple terms, we are training a machine to learn how macroeconomic factors have influenced currency movements in the past. Then we use that machine to simulate how future macro conditions might affect FX rates.

    The goal is not just to predict the future, but to explore “what-if” scenarios and better understand the link between economic fundamentals and global currency markets.
    """)


with tab6:
    st.markdown("## Scenario Simulator")
    st.write("Simulate hypothetical changes in U.S. macro variables and forecast FX rates from 1 to 20 years.")

    st.markdown("### 1. Select Forecast Settings")

    currency_options = [
        "USD-EUR", "USD-JPY", "USD-GBP", "USD-CHF", "USD-CAD",
        "USD-AUD", "USD-NZD", "USD-CNY", "USD-HKD", "USD-XAU"
    ]
    selected_currencies = st.multiselect("Select Currency Pairs", options=currency_options, default=["USD-EUR"])

    macro_options = [
        'Interest Rate', 'Inflation (CPI)', 'Core Inflation', 'Industrial Production',
        'Trade Balance', 'Unemployment Rate', 'Consumer Sentiment', 'Retail Sales',
        'Manufacturing PMI', 'S&P 500 Index', 'VIX Index'
    ]
    selected_macros = st.multiselect("Select Macroeconomic Indicators", options=macro_options, default=['Interest Rate', 'Inflation (CPI)'])

    model_choice = st.selectbox("Select Model", ["OLS", "Lasso", "LassoCV", "XGBoost"])
    forecast_years = st.number_input("Select Forecast Horizon - Years (Enter a number between 1 to 20)", min_value=1, max_value=20, value=5, step=1)

    st.markdown("---")
    st.markdown("### 2. Adjust Macroeconomic Scenario")

    macro_adjustments = {}
    for macro in selected_macros:
        adj = st.slider(f"Adjust %s (%%)" % macro, -5.0, 5.0, 0.5, step=0.1)
        macro_adjustments[macro] = adj

    st.markdown("---")
    use_log_scale = st.checkbox("Use Log Scale for Forecast Graph", value=True)

    if st.button("Run Simulation"):
        from run_forex_model import run_forex_model
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        real_metrics, future_predictions, test_prediction_df = run_forex_model(
            selected_currencies=selected_currencies,
            selected_macros=selected_macros,
            selected_model=model_choice,
            macro_adjustments=macro_adjustments,
            future_years=forecast_years
        )

        st.session_state['real_metrics'] = real_metrics
        st.session_state['future_predictions'] = future_predictions
        st.session_state['test_prediction_df'] = test_prediction_df

        st.success("Simulation completed. Forecast and correlation matrices available below.")

    if 'future_predictions' in st.session_state:
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd

        future_predictions = st.session_state['future_predictions']
        real_metrics = st.session_state['real_metrics']
        test_prediction_df = st.session_state['test_prediction_df']

        st.markdown("### Model Performance on Real 2023–2025 Data")
        st.dataframe(real_metrics)

        st.markdown("### Actual vs Predicted Returns (2023–2025)")
        for i in range(0, len(selected_currencies), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(selected_currencies):
                    currency = selected_currencies[i + j]
                    df = test_prediction_df[test_prediction_df['Currency'] == currency]
                    with cols[j]:
                        st.markdown(f"**{currency}: Actual vs Predicted Log Returns**")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df['DATE'], y=df['Actual_Return'], mode='lines', name='Actual', line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=df['DATE'], y=df['Predicted_Return'], mode='lines', name='Predicted', line=dict(color='orange')))
                        fig.update_layout(xaxis_title="Date", yaxis_title="Log Return", height=300, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### Forecasted Forex Rates for Next {forecast_years} Years")
        fig = go.Figure()
        for currency in selected_currencies:
            df = future_predictions[future_predictions['Currency'] == currency]
            fig.add_trace(go.Scatter(
                x=df['DATE'],
                y=df['Predicted_Forex_Rate'],
                mode='lines+markers',
                name=currency,
                line=dict(width=2),
                marker=dict(size=4),
                hovertemplate=("<b>%s</b><br>Date: %%{x|%%b %%Y}<br>Rate: %%{y:.4f}<extra></extra>" % currency)
            ))
        if use_log_scale:
            fig.update_yaxes(type="log", title="Predicted Forex Rate (Log Scale)")
            fig.update_layout(title="Forecasted Forex Rates (Log Scale)")
        else:
            fig.update_yaxes(title="Predicted Forex Rate (Linear Scale)")
            fig.update_layout(title="Forecasted Forex Rates (Linear Scale)")

        fig.update_layout(
            xaxis_title="Date",
            legend_title="Currency",
            hovermode="x unified",
            template="plotly_white",
            height=450,
            margin=dict(t=50, b=30, l=50, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Download Forecasted Future Predictions")
        future_csv = future_predictions.to_csv(index=False)
        st.download_button("Download Future Predictions CSV", data=future_csv, file_name='future_predictions.csv', mime='text/csv')

        st.markdown("### Correlation Matrices")
        macro_raw = pd.read_csv("macro_data.csv", parse_dates=["DATE"]).set_index("DATE")
        forex_raw = pd.read_csv("forex_merged_cleaned.csv", parse_dates=["DATE"]).set_index("DATE")

        macro_selected = macro_raw[selected_macros].dropna()
        macro_corr_matrix = macro_selected.corr()

        forex_log_returns = np.log(forex_raw / forex_raw.shift(1)).dropna()
        combined_df = pd.merge(macro_selected, forex_log_returns[selected_currencies], left_index=True, right_index=True).dropna()
        macro_to_forex_corr = combined_df.corr().loc[selected_macros, selected_currencies]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Macro-Macro Correlation (Interactive)")
            fig_corr_macro = go.Figure(
                data=go.Heatmap(
                    z=macro_corr_matrix.values,
                    x=macro_corr_matrix.columns,
                    y=macro_corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Corr"),
                    hovertemplate='Macro 1: %{y}<br>Macro 2: %{x}<br>Corr: %{z:.2f}<extra></extra>'
                )
            )
            fig_corr_macro.update_layout(title="Macro ↔ Macro Correlation", height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig_corr_macro, use_container_width=True)

        with col2:
            st.markdown("#### Macro-Forex Correlation (Interactive)")
            fig_corr_macrofx = go.Figure(
                data=go.Heatmap(
                    z=macro_to_forex_corr.values,
                    x=macro_to_forex_corr.columns,
                    y=macro_to_forex_corr.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Corr"),
                    hovertemplate='Macro: %{y}<br>Currency: %{x}<br>Corr: %{z:.2f}<extra></extra>'
                )
            )
            fig_corr_macrofx.update_layout(title="Macro ↔ Forex Correlation", height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig_corr_macrofx, use_container_width=True)

        st.markdown("### Download Correlation Matrices")
        macro_corr_csv = macro_corr_matrix.to_csv()
        macro_forex_corr_csv = macro_to_forex_corr.to_csv()

        st.download_button("Download Macro-Macro Correlation CSV", data=macro_corr_csv, file_name='macro_correlation_matrix.csv', mime='text/csv')
        st.download_button("Download Macro-Forex Correlation CSV", data=macro_forex_corr_csv, file_name='macro_forex_correlation_matrix.csv', mime='text/csv')
