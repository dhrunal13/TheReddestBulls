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
    try:
        with open("macro_desc.md", "r", encoding='utf-8') as file:
            macro_desc_text = file.read()
        st.markdown(macro_desc_text)
    except FileNotFoundError:
        st.error("Macro description file not found.")

    st.markdown("### Forex Rates Description")
    try:
        with open("forex_desc.md", "r", encoding='utf-8') as file:
            forex_desc_text = file.read()
        st.markdown(forex_desc_text)
    except FileNotFoundError:
        st.error("Forex description file not found.")

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

    Features coming soon:
    - Coefficient tables
    - P-values and significance highlights
    - Cross-currency comparisons
    """)

with tab5:
    st.markdown("## Forecasting Models")
    st.write("""
    Compare predictive model performance:
    
    - OLS Regression
    - Lasso Regression
    - LassoCV Regression
    - XGBoost Regression

    Metrics to be compared:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
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
