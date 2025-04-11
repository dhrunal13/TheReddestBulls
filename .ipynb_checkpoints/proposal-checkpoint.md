# Project Proposal: Macro Drivers of Dollar Strength - A Cross-Currency Analysis 
By - The Reddest Bulls


## Research Question

### Big Picture Question  
How do a country’s macroeconomic fundamentals influence its currency's value in the global market? Specifically, how do shifts in the **U.S. macroeconomy** impact the strength of the **U.S. dollar** relative to other major global currencies?

### Specific Research Question  
How do **monthly changes in U.S. macroeconomic indicators** affect the **value of the U.S. dollar** relative to the top 10 most-traded currencies (excluding the USD itself)?

We also explore whether certain currencies are **more sensitive** to specific U.S. macro variables than others. Are commodity currencies (like AUD or CAD) more affected by trade balances? Are European currencies more interest-rate sensitive?

### Hypotheses  
- **H1:** A rise in U.S. interest rates is associated with dollar appreciation against most major currencies.  
- **H2:** Higher U.S. inflation is associated with short-term dollar depreciation.  
- **H3:** The dollar appreciates when U.S. trade deficits narrow (or surpluses widen).  
- **H4 (extension):** Currency responses to U.S. macro shocks vary significantly across currencies based on their economic structure and trade relationships.

### Metrics of Success  
This is primarily a **predictive regression** project, with interpretability and forecasting accuracy as key goals:
- We will use **Ordinary Least Squares (OLS)** as the primary model to predict monthly changes in currency value based on U.S. macro data
- Then measure performance using **out-of-sample RMSE, MAE**, and visual inspection of predicted vs. actual trends
- Later track **coefficient stability and significance** across currencies to compare macro sensitivity
- Finally we have it forecasts for 2023–present and a dashboard to simulate macroeconomic scenarios
  
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
- **Investing.com**, **Yahoo Finance**, or **FRED** for exchange rates  
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
