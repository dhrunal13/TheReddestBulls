# Project Proposal: Macro Drivers of Dollar Strength - A Cross-Currency Analysis 
By - The Reddest Bulls

## Research Question

### Big Picture Question  
How do a country’s macroeconomic fundamentals influence its currency's value in the global market? Specifically, how do shifts in the **U.S. macroeconomy** impact the strength of the **U.S. dollar** relative to other major global currencies?

### Specific Research Question  
How do **monthly changes in U.S. macroeconomic indicators** affect the **value of the U.S. dollar** relative to the top 10 most-traded currencies (excluding the USD itself)?

We also explore whether certain currencies are **more sensitive** to specific U.S. macro variables than others. Are commodity currencies (like AUD or CAD) more affected by trade balances? Are European currencies more interest-rate sensitive?

## Hypotheses  
- **H1:** A rise in U.S. interest rates is associated with dollar appreciation against most major currencies.  
- **H2:** Higher U.S. inflation is associated with short-term dollar depreciation.  
- **H3:** The dollar appreciates when U.S. trade deficits narrow (or surpluses widen).  
- **H4 (extension):** Currency responses to U.S. macro shocks vary significantly across currencies based on their economic structure and trade relationships.

## Theoretical Motivation: Linking Macro Indicators to FX Movements

The value of a currency in foreign exchange markets is fundamentally linked to macroeconomic conditions via several international finance theories:

- **Interest Rate Parity (IRP):** Higher U.S. interest rates attract global capital inflows, driving up demand for USD assets and appreciating the dollar (**H1**).
- **Purchasing Power Parity (PPP):** Inflation differentials reduce the real value of a currency; high U.S. inflation weakens the dollar relative to more stable currencies (**H2**).
- **Balance of Payments Theory:** An improving trade balance increases demand for USD by reducing net outflows (**H3**).
- **Heterogeneous Sensitivity Hypothesis:** Currency sensitivity depends on economic structure — e.g., trade dependency, monetary regime, or commodity reliance (**H4**).

## Methodology & Evaluation  
This is primarily a **predictive regression project** with emphasis on interpretability and comparative macro sensitivity across currencies. Key steps:

- Use **OLS** as the main model to estimate macroeconomic sensitivity per currency.
- Introduce a **second model** — such as **Lasso Regression** or **XGBoost** — to find the best prediction performance.
- Evaluate using:
  - **Out-of-sample RMSE & MAE**
  - **Visual accuracy:** predicted vs actual FX trends
  - **Significance & direction of coefficients** per macro indicator
  - **Cross-currency comparison of sensitivity** via clustering

## Dashboard Design
The final product will include an interactive dashboard to simulate, visualize, and interpret how macro variables impact dollar strength across currencies.

### Dashboard Tabs

- **Overview:** Project summary and motivation
- **Hypotheses:** Statement of hypotheses with links to supporting visuals
- **Data Explorer:** Raw FX and macro data visualized over time
- **Static Analysis:** OLS regression results (per-currency coefficients, p-values)
- **Forecasting Models:** Compare OLS and best-performing model (Lasso/XGBoost)
- **Scenario Simulator:** Simulate macro changes and predict FX impacts

### Interactivity Features

- **Dropdown 1:** Currency selector (EUR/USD, JPY/USD, etc.)
- **Dropdown 2:** Macroeconomic indicator selector (e.g., interest rate, inflation)
- **Date Range Slider:** Custom time selection
- **Scenario Slider:** Adjust macro input (e.g., +0.5% interest rate)
- **Model Toggle:** Switch between OLS and Lasso/XGBoost output
- **Hypothesis Shortcut Buttons:** Show dashboard output directly relevant to each hypothesis

## Data Info

### Variables Needed

**Dependent Variable:**  
- Monthly % change in each currency’s value relative to USD (log return or percent change)

**Independent Variables (U.S. macroeconomic indicators):**  
- **Interest Rate:** Effective Federal Funds Rate  
- **Inflation:** CPI, Core CPI  
- **GDP Growth:** Industrial Production Index (monthly proxy)  
- **Trade Balance:** Monthly U.S. trade balance (BEA)  
- **Unemployment Rate:** BLS unemployment rate  
- **Consumer Sentiment:** U. Michigan Consumer Sentiment Index  
- **Retail Sales:** Monthly U.S. retail sales growth  
- **Manufacturing Activity:** ISM Manufacturing PMI  
- **Stock Market Performance:** % change in S&P 500 (optional)  
- **Volatility/Uncertainty:** VIX Index, Economic Policy Uncertainty Index (optional controls)

### Data Sources  
- **FRED** for U.S. macro data  
- **Bloomberg Terminal** for exchange rates  
- **OurWorldInData** for backup macro validation

### Collection & Structure  
- Store raw data under `/raw_data/` organized by source  
- Merge monthly macro indicators into one time-indexed panel  
- Convert FX rates into monthly log returns or percent changes  
- Store cleaned dataset as `/processed_data/usd_forex_panel.csv`

### Data Transformation (High Level)  
- Normalize or standardize macro variables  
- Lag predictors to test for delayed effects  
- Train OLS models for each currency  
- Evaluate forecasts using 2023–present test set  
- Cluster or rank currencies by macro sensitivity (e.g., interest-sensitive vs inflation-sensitive)  
- Split data:
  - **Train:** Jan 2002 – Dec 2022  
  - **Test:** Jan 2023 – Present  
- Build dashboard for users to explore FX behavior by currency and macro driver