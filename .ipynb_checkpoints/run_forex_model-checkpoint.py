import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm


# ---------------------------------------
# Load and prepare data
# ---------------------------------------
def load_and_prepare_data():
    macro = pd.read_csv("macro_data.csv", parse_dates=["DATE"]).set_index("DATE")
    forex = pd.read_csv("forex_merged_cleaned.csv", parse_dates=["DATE"]).set_index("DATE")

    log_returns = np.log(forex / forex.shift(1)).dropna()
    log_returns.columns = [col + " Return" for col in log_returns.columns]

    LAG_PERIODS = (1, 2, 3, 4, 5, 12, 24, 60)
    ROLL_WINDOWS = (3, 5, 7, 10)

    def enrich_macro_safe(df):
        fea = df.copy()
        for p in LAG_PERIODS:
            fea = pd.concat([fea, df.shift(p).add_suffix(f"_lag{p}")], axis=1)
        for w in ROLL_WINDOWS:
            fea = pd.concat([
                fea,
                df.rolling(w).mean().shift(1).add_suffix(f"_rollmean{w}"),
                df.rolling(w).std().shift(1).add_suffix(f"_rollstd{w}")
            ], axis=1)
        return fea.dropna()

    macro_fea = enrich_macro_safe(macro)
    full_df = pd.merge(macro_fea, log_returns, left_index=True, right_index=True).dropna()
    return full_df, forex

# ---------------------------------------
# Model training helper
# ---------------------------------------
def train_model(X_train, y_train, selected_model):
    if selected_model == 'Lasso':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = Lasso(alpha=0.01, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        return model, scaler
    elif selected_model == 'XGBoost':
        model = XGBRegressor(n_estimators=500, learning_rate=0.05)
        model.fit(X_train, y_train)
        return model, None
    elif selected_model == 'OLS':
        X_train_const = sm.add_constant(X_train, has_constant='add')
        model = sm.OLS(y_train, X_train_const).fit()
        return model, None
    elif selected_model == 'LassoCV':
        transformer = Pipeline([('scaler', StandardScaler()),
                                 ('pca', PCA(n_components = 0.975)) # min components needed to explain 97.5% of the variance in data
                               ])
        X_reduced = transformer.fit_transform(X_train)
        model = LassoCV(cv = TimeSeriesSplit(n_splits = 5), # randomized sections that respect time path
                        max_iter = 10000,
                        tol = 1e-3,
                       )
        model.fit(X_reduced, y_train)
            
        return model, transformer
    else:
        raise ValueError(f"Model {selected_model} not supported.")

# ---------------------------------------
# Main function to run model(s)
# ---------------------------------------
def run_forex_model(
    selected_currencies,
    selected_macros,
    selected_model,
    macro_adjustments=None,
    future_years=10
):
    full_df, forex = load_and_prepare_data()
    all_real_metrics = []
    all_future_preds = []
    all_test_predictions = []  # NEW: to store actual vs predicted for 2023–2025

    for selected_currency in selected_currencies:
        target_column = selected_currency + " Return"

        if target_column not in full_df.columns:
            print(f"Skipping {selected_currency}: not found in data.")
            continue

        available_macros = [col for col in full_df.columns if any(macro in col for macro in selected_macros)]

        train_end = "2022-12-31"
        train_df = full_df[full_df.index <= train_end]
        test_df = full_df[full_df.index > train_end]

        y_train = train_df[target_column]
        y_test = test_df[target_column]
        X_train = train_df[available_macros]
        X_test = test_df[available_macros]

        model, scaler = train_model(X_train, y_train, selected_model)

        # Predict test
        if selected_model == 'OLS':
            X_test_const = sm.add_constant(X_test, has_constant='add')
            preds = model.predict(X_test_const)
        else:
            preds = model.predict(scaler.transform(X_test)) if scaler else model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        all_real_metrics.append({
            "Currency": selected_currency,
            "Model": selected_model,
            "Selected_Macros": selected_macros,
            "R2_Score": r2,
            "MAE": mae,
            "RMSE": rmse,
        })

        # NEW: Store actual vs predicted test set results
        test_df_comp = pd.DataFrame({
            'DATE': y_test.index,
            'Currency': selected_currency,
            'Actual_Return': y_test.values,
            'Predicted_Return': preds
        })
        all_test_predictions.append(test_df_comp)

        # Predict future
        last_known_macros = train_df[available_macros].iloc[-1]
        months = future_years * 12
        future_macro_df = pd.DataFrame([last_known_macros.values] * months, columns=available_macros)

        if macro_adjustments:
            for macro, adjustment in macro_adjustments.items():
                matching_cols = [col for col in available_macros if macro in col]
                for col in matching_cols:
                    future_macro_df[col] += adjustment

        if selected_model == 'OLS':
            future_macro_const = sm.add_constant(future_macro_df, has_constant='add')
            future_preds = model.predict(future_macro_const)
        else:
            future_preds = model.predict(scaler.transform(future_macro_df)) if scaler else model.predict(future_macro_df)

        last_real_price = forex[selected_currency].iloc[-1]
        future_price_series = last_real_price * np.exp(pd.Series(future_preds)).cumprod()

        future_dates = pd.date_range(start="2025-01-01", periods=months, freq='MS')
        temp_future_df = pd.DataFrame({
            'DATE': future_dates,
            'Currency': selected_currency,
            'Predicted_Return': future_preds,
            'Predicted_Forex_Rate': future_price_series.values
        })

        all_future_preds.append(temp_future_df)

    real_metrics_df = pd.DataFrame(all_real_metrics)
    future_predictions_df = pd.concat(all_future_preds)
    test_prediction_df = pd.concat(all_test_predictions) if all_test_predictions else pd.DataFrame()

    return real_metrics_df, future_predictions_df, test_prediction_df
