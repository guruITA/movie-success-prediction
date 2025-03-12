import os
import json
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Settings & Database Connection
# ---------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
conn_url = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(conn_url)

# ---------------------------
# Read CSV and Display Basic Info
# ---------------------------
csv_file = Path("data/IMDb movies.csv")
df = pd.read_csv(csv_file)

# ---------------------------
# Financial Data: Extraction & Conversion to USD
# ---------------------------
financial_cols = ["budget", "usa_gross_income", "worlwide_gross_income"]  # List of financial columns

for col in financial_cols:                # Loop through each financial column
    # Extract numeric value from the column (e.g., "1000000") and convert to numeric type
    df[f"{col}_value"] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    # Extract the currency symbol from the column (e.g., "$")
    df[f"{col}_currency"] = df[col].astype(str).str.extract(r'([^\d\s]+)')[0]

def convert_to_usd(value, currency):      # Define function to convert a value to USD
    rates = {                           # Dictionary of exchange rates
        "$": 1.0, "ROL": 0.20742, "ITL": 0.00054, "SEK": 0.096, "FRF": 0.153, "NOK": 0.096,
        "GBP": 1.27, "DEM": 0.568, "PTE": 0.006, "CZK": 0.044, "INR": 0.012, "CAD": 0.79,
        "CHF": 1.09, "FIM": 0.168, "JPY": 0.0072, "BEF": 0.025, "NLG": 0.51, "DKK": 0.14,
        "KRW": 0.00076, "PLN": 0.26, "RUR": 0.011, "AUD": 0.66, "TRL": 0.000051, "NZD": 0.62,
        "HKD": 0.13, "EGP": 0.032, "ESP": 0.006, "CNY": 0.14, "EUR": 1.08, "ALL": 0.0096,
        "ISK": 0.0072, "DOP": 0.017, "BDT": 0.0092, "IEP": 1.27, "IRR": 0.000024, "HRK": 0.14,
        "JMD": 0.0065, "SIT": 0.0045, "PHP": 0.018, "SGD": 0.73, "BRL": 0.20, "HUF": 0.0029,
        "EEK": 0.068, "BND": 0.73, "THB": 0.029, "GRD": 0.003, "ZAR": 0.053, "ATS": 0.078,
        "IDR": 0.000066, "PKR": 0.0036, "CLP": 0.0012, "BGL": 0.56, "NPR": 0.0075, "MXN": 0.059,
        "YUM": 0.0096, "LTL": 0.34, "MYR": 0.21, "PYG": 0.00014, "SKK": 0.034, "MTL": 2.58,
        "TWD": 0.031, "UAH": 0.027, "ARS": 0.0023, "COP": 0.00025, "LVL": 1.62, "VND": 0.000041,
        "NGN": 0.0013, "VEB": 0.00010, "ILS": 0.27, "TTD": 0.15, "GEL": 0.38, "AZM": 0.00012,
        "RON": 0.21, "LKR": 0.0032, "MNT": 0.00029, "AMD": 0.0026, "AED": 0.27
    }
    if pd.isna(value) or pd.isna(currency):  # If value or currency is missing, return the original value
        return value
    return value * rates.get(currency, 0.0)  # Multiply value by the corresponding exchange rate

for col in financial_cols:                # Apply the conversion function to each financial column
    df[col] = df.apply(lambda r: convert_to_usd(r[f"{col}_value"], r[f"{col}_currency"]), axis=1)

# ---------------------------
# Data Cleaning
# ---------------------------
df.replace(0.0, pd.NA, inplace=True)
df = df.dropna(subset=["budget"])  # Ensure budget is available
df = df.dropna(subset=["avg_vote"])  # Ensure IMDb rating is available
df = df[df["budget"] >= 1000]  # Filter out unrealistic budgets

# ---------------------------
# Feature Encoding
# ---------------------------
categorical_cols = ['director', 'actors', 'genre', 'country', 'language']
mapping = {col: defaultdict(float) for col in categorical_cols}
counts = {col: defaultdict(int) for col in categorical_cols}

for idx, row in df.iterrows():
    target_val = row["avg_vote"] # Use IMDb rating as target
    for col in categorical_cols:
        if pd.isna(row[col]):
            continue
        tokens = [t.strip() for t in str(row[col]).split(',')]
        for token in tokens:
            mapping[col][token] += target_val
            counts[col][token] += 1

target_mapping = {
    col: {token: mapping[col][token] / counts[col][token] for token in mapping[col]}
    for col in categorical_cols
}

def encode_features(row):
    num_feature = np.log(float(row["budget"]) + 1)
    cat_features = []
    for col in categorical_cols:
        if pd.isna(row[col]):
            cat_features.append(0.0)
        else:
            tokens = [t.strip() for t in str(row[col]).split(',')]
            values = [target_mapping[col].get(token, 0.0) for token in tokens]
            cat_features.append(np.mean(values))
    return [num_feature] + cat_features

df["encoded_features"] = df.apply(lambda row: encode_features(row), axis=1)

# ---------------------------
# Model Training
# ---------------------------
X = np.vstack(df["encoded_features"].tolist())
y = df["avg_vote"].values  # Now predicting IMDb rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train SGD Regressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)

# ---------------------------
# Model Evaluation
# ---------------------------
print("Linear Regression:")
print("R^2:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

print("\nRandom Forest:")
print("R^2:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

print("\nSGD Regressor:")
print("R^2:", r2_score(y_test, y_pred_sgd))
print("MSE:", mean_squared_error(y_test, y_pred_sgd))

# ---------------------------
# Visualization
# ---------------------------
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf
residuals_sgd = y_test - y_pred_sgd

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Linear Regression
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[0, 0].set_title("Linear Regression: Predicted vs Actual")
axes[0, 1].hist(residuals_lr, bins=30, edgecolor='k', alpha=0.7)
axes[0, 1].set_title("Linear Regression: Residuals")

# Random Forest
axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.5, color="green")
axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[1, 0].set_title("Random Forest: Predicted vs Actual")
axes[1, 1].hist(residuals_rf, bins=30, edgecolor='k', alpha=0.7, color="green")
axes[1, 1].set_title("Random Forest: Residuals")

# SGD Regressor
axes[2, 0].scatter(y_test, y_pred_sgd, alpha=0.5, color="orange")
axes[2, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
axes[2, 0].set_title("SGD Regressor: Predicted vs Actual")
axes[2, 1].hist(residuals_sgd, bins=30, edgecolor='k', alpha=0.7, color="orange")
axes[2, 1].set_title("SGD Regressor: Residuals")

plt.tight_layout()
plt.show()

print("Script completed.")
