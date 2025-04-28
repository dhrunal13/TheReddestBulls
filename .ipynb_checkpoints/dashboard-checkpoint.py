# dashboard.py

import streamlit as st

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
st.markdown("<h1 style='text-align: center; color: #C41E3A;'>The Reddest Bulls</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Macroeconomic Drivers of FX Rates</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #C41E3A'>", unsafe_allow_html=True)

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
    st.markdown("## Data Explorer")
    st.write("""
    Explore raw macroeconomic indicators and FX rate trends over time.

    Features coming soon:
    - Dropdown selectors for currency and macro variables
    - Interactive time series plots
    - Customizable date ranges
    """)

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
    - XGBoost Regression

    Metrics to be compared:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    """)

with tab6:
    st.markdown("## Scenario Simulator")
    st.write("""
    Simulate the effects of hypothetical macroeconomic changes on FX rates.

    Planned Features:
    - Currency and macroeconomic indicator selectors
    - Adjustable macro parameter sliders
    - Model choice toggle (OLS vs Lasso/XGBoost)
    - Forecasted FX rate impact visualization
    """)

