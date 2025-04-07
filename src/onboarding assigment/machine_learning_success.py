import os                              # Import os module for operating system functions
import json                            # Import json module for handling JSON data
import logging                         # Import logging module to log messages
import numpy as np                     # Import numpy for numerical operations
import pandas as pd                    # Import pandas for data manipulation
from sqlalchemy import create_engine, text  # Import create_engine and text from SQLAlchemy for DB connection and queries
from dotenv import load_dotenv         # Import load_dotenv to load environment variables from .env file
import matplotlib.pyplot as plt        # Import matplotlib for plotting graphs
from pathlib import Path               # Import Path from pathlib for file path handling
from collections import defaultdict    # Import defaultdict to create dictionaries with default values

# ---------------------------
# Settings & Database Connection
# ---------------------------
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
load_dotenv()                             # Load environment variables from .env file
db_user = os.getenv("DB_USER")            # Get database username
db_pass = os.getenv("DB_PASSWORD")        # Get database password
db_host = os.getenv("DB_HOST")            # Get database host
db_port = os.getenv("DB_PORT")            # Get database port
db_name = os.getenv("DB_NAME")            # Get database name
conn_url = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"  # Build the connection URL
engine = create_engine(conn_url)          # Create the SQLAlchemy engine for the DB connection

# ---------------------------
# Read CSV and Display Basic Info
# ---------------------------
csv_file = Path("data/IMDb movies.csv")   # Specify the path to the CSV file
df = pd.read_csv(csv_file)                # Read the CSV file into a DataFrame

print("<------------------------>")      
print("First 5 rows:")                    
print(df.head(5))                         # Print the first 5 rows of the DataFrame
print("<------------------------>")
print("Dataset info:")
df.info()                                 # Display dataset information (columns, data types, etc.)
print("<------------------------>")

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
# Data Cleaning & Calculating Success Score
# ---------------------------
df.replace(0.0, pd.NA, inplace=True)          # Replace 0.0 values with NA
df = df.dropna(subset=["budget_value"])       # Drop rows where budget value is missing
df = df.dropna(subset=["usa_gross_income", "worlwide_gross_income"], how="all")  # Drop rows missing both income fields
df["corrected_income"] = df["worlwide_gross_income"].fillna(df["usa_gross_income"])  # Use worldwide income if available
df = df.dropna(subset=["avg_vote"])           # Drop rows with no average vote
df = df[df["budget_value"] >= 1000]           # Keep only movies with a significant budget (>= 1000)

# Calculate financial success (ROI) and vote success (normalized vote)
df["financial_success"] = (df["corrected_income"] - df["budget_value"]) / df["budget_value"]
df["vote_success"] = df["avg_vote"] / 10       # Normalize average vote by dividing by 10
# Calculate composite success as weighted sum (70% financial, 30% vote)
df["composite_success"] = 0.7 * df["financial_success"] + 0.3 * df["vote_success"]
# Scale composite success using tanh to reduce extreme values
df["composite_success_scaled"] = np.tanh(df["composite_success"].astype(np.float64))

print("<------------------------>")
print("Example Success Score:")           # Print an example of the calculated success score
print(df[["title", "budget", "corrected_income", "composite_success_scaled"]].head(5))
print("<------------------------>")

# ---------------------------
# Target Encoding for Categorical Features
# ---------------------------
# Define the categorical columns for which target encoding will be applied.
categorical_cols = ['director', 'actors', 'genre', 'country', 'language']

# Create a dictionary for each column to store the sum of composite success scores for each token.
# For example, mapping["actors"]["Actor A"] will accumulate the composite success scores for "Actor A".
mapping = {col: defaultdict(float) for col in categorical_cols}

# Create a dictionary for each column to count the number of times each token appears.
# For example, counts["actors"]["Actor A"] will count how many times "Actor A" appears.
counts = {col: defaultdict(int) for col in categorical_cols}

# Loop over each row (film) in the DataFrame.
for idx, row in df.iterrows():
    # Retrieve the composite success score (scaled) for the current film.
    target_val = row["composite_success_scaled"]
    # Loop through each of the defined categorical columns.
    for col in categorical_cols:
        # If the column value is missing, skip processing for this column.
        if pd.isna(row[col]):
            continue
        # Split the string by comma to get individual tokens.
        # For example, "Actor A, Actor B" becomes ["Actor A", "Actor B"].
        tokens = [t.strip() for t in str(row[col]).split(',')]
        # For each token, update the total composite score and increment its occurrence count.
        for token in tokens:
            mapping[col][token] += target_val  # Add the film's composite score to the token's total score.
            counts[col][token] += 1              # Increment the token's count by one.

# Create the target mapping by computing the average composite success score for each token.
target_mapping = {}
for col in categorical_cols:
    target_mapping[col] = {}  # Initialize an empty dictionary for this column.
    for token in mapping[col]:
        # Compute the average score for the token by dividing the total score by the count.
        target_mapping[col][token] = mapping[col][token] / counts[col][token]

# ---------------------------
# Feature Encoding: Log-Transformed Budget + Target Encoded Features
# ---------------------------
def encode_features(row):
    # Apply logarithmic transformation to the budget to reduce skewness and limit the influence of outliers.
    # Adding 1 avoids issues with log(0).
    num_feature = np.log(float(row["budget"]) + 1)
    
    # Initialize an empty list to hold the encoded values for each categorical feature.
    cat_features = []
    
    # Loop through each categorical column defined in categorical_cols.
    for col in categorical_cols:
        # If the value in this column is missing, append 0.0 as the encoded value.
        if pd.isna(row[col]):
            cat_features.append(0.0)
        else:
            # Split the column's string by commas to get individual tokens (e.g., "Actor A, Actor B" becomes ["Actor A", "Actor B"]).
            tokens = [t.strip() for t in str(row[col]).split(',')]
            # For each token, retrieve its target encoded value from target_mapping.
            # If a token is not found, default to 0.0.
            values = [target_mapping[col].get(token, 0.0) for token in tokens]
            # Compute the average of the target encoded values for the column.
            cat_features.append(np.mean(values))
    
    # Return a combined feature vector:
    # The first element is the log-transformed budget,
    # followed by the averaged target encoded values for each categorical column.
    return [num_feature] + cat_features

print("Encoding features...")  # Indicate that feature encoding is in progress.
# Apply the encode_features function to each row in the DataFrame and store the result in a new column 'encoded_features'.
df["encoded_features"] = df.apply(lambda row: encode_features(row), axis=1)
print("Examples:")  # Print examples of the encoded features.
print(df[["title", "encoded_features"]].head(5))  # Display the first 5 rows showing the title and its corresponding encoded feature vector.

# ---------------------------
# Upload Processed Data to the Database
# ---------------------------
table_name = "movie_prediction_features"          # Define the database table name
with engine.connect() as conn:                    # Connect to the database
    conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))  # Drop table if it exists
    conn.execute(text(f"""
        CREATE TABLE {table_name} (
            imdb_title_id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(255),
            features JSON,
            target_success FLOAT
        );
    """))                                       # Create a new table with required columns
    conn.commit()                               # Commit the changes

print("Uploading data to database...")           # Print upload status
upload_df = pd.DataFrame({                       # Create a DataFrame with columns for upload
    "imdb_title_id": df["imdb_title_id"],
    "title": df["title"],
    "features": df["encoded_features"].apply(lambda x: json.dumps(x)),  # Convert features to JSON format
    "target_success": df["composite_success_scaled"]
})
upload_df.to_sql(table_name, con=engine, if_exists='append', index=False)  # Upload the data to the database table
print("Data uploaded.")                          # Confirm data upload

# ---------------------------
# Model Training: Retrieve Data and Train Models
# ---------------------------
print("Retrieving data for model training...")   # Print retrieval status
query = f"SELECT imdb_title_id, title, features, target_success FROM {table_name}"  # Define SQL query to retrieve data
model_df = pd.read_sql(query, con=engine)         # Read data from database into a DataFrame
model_df["features"] = model_df["features"].apply(json.loads)  # Decode the JSON features back to Python objects
X = np.vstack(model_df["features"].tolist())      # Convert list of feature vectors into a NumPy array
y = model_df["target_success"].values             # Extract target values

from sklearn.model_selection import train_test_split  # Import train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into train and test sets

# Train Linear Regression Model
from sklearn.linear_model import LinearRegression      # Import LinearRegression model
lr_model = LinearRegression()                          # Initialize the Linear Regression model
lr_model.fit(X_train, y_train)                         # Fit the model on training data
y_pred_lr = lr_model.predict(X_test)                   # Predict on the test set
print("Linear Regression Performance:")                # Print performance metrics for Linear Regression
from sklearn.metrics import r2_score, mean_squared_error  # Import evaluation metrics
print("R^2:", r2_score(y_test, y_pred_lr))             # Print R^2 score
print("MSE:", mean_squared_error(y_test, y_pred_lr))   # Print Mean Squared Error

# Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor    # Import RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initialize the Random Forest model
rf_model.fit(X_train, y_train)                        # Fit the model on training data
y_pred_rf = rf_model.predict(X_test)                  # Predict on the test set
print("Random Forest Performance:")                   # Print performance metrics for Random Forest
print("R^2:", r2_score(y_test, y_pred_rf))            # Print R^2 score
print("MSE:", mean_squared_error(y_test, y_pred_rf))  # Print Mean Squared Error

# Train SGD Regressor
from sklearn.linear_model import SGDRegressor          # Import SGDRegressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # Initialize the SGD Regressor
sgd_model.fit(X_train, y_train)                        # Fit the SGD model on training data
y_pred_sgd = sgd_model.predict(X_test)                 # Predict on the test set
print("SGD Regressor Performance:")                    # Print performance metrics for SGD Regressor
print("R^2:", r2_score(y_test, y_pred_sgd))            # Print R^2 score
print("MSE:", mean_squared_error(y_test, y_pred_sgd))  # Print Mean Squared Error

# ---------------------------
# Combined Visualization (3×2)
# ---------------------------
# Calculate residuals for each model (actual - predicted)
residuals_lr = y_test - y_pred_lr                    # Residuals for Linear Regression
residuals_rf = y_test - y_pred_rf                    # Residuals for Random Forest
residuals_sgd = y_test - y_pred_sgd                  # Residuals for SGD Regressor

# Create a figure with 3 rows and 2 columns (3 models, each with 2 plots)
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# Row 1: Linear Regression
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5)     # Scatter plot: Actual vs Predicted (Linear Regression)
axes[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Ideal line
axes[0, 0].set_title("Linear Regression: Predicted vs Actual")  # Set title
axes[0, 0].set_xlabel("Actual Composite Success (Scaled)")      # Set x-axis label
axes[0, 0].set_ylabel("Predicted Composite Success (Scaled)")   # Set y-axis label

axes[0, 1].hist(residuals_lr, bins=30, edgecolor='k', alpha=0.7)  # Histogram of residuals (Linear Regression)
axes[0, 1].set_title("Linear Regression: Residual Distribution")  # Set title
axes[0, 1].set_xlabel("Residuals")                                # Set x-axis label
axes[0, 1].set_ylabel("Frequency")                                # Set y-axis label

# Row 2: Random Forest Regressor
axes[1, 0].scatter(y_test, y_pred_rf, alpha=0.5, color="green")  # Scatter plot: Actual vs Predicted (Random Forest)
axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Ideal line
axes[1, 0].set_title("Random Forest: Predicted vs Actual")         # Set title
axes[1, 0].set_xlabel("Actual Composite Success (Scaled)")         # Set x-axis label
axes[1, 0].set_ylabel("Predicted Composite Success (Scaled)")      # Set y-axis label

axes[1, 1].hist(residuals_rf, bins=30, edgecolor='k', alpha=0.7, color="green")  # Histogram of residuals (Random Forest)
axes[1, 1].set_title("Random Forest: Residual Distribution")        # Set title
axes[1, 1].set_xlabel("Residuals")                                  # Set x-axis label
axes[1, 1].set_ylabel("Frequency")                                  # Set y-axis label

# Row 3: SGD Regressor
axes[2, 0].scatter(y_test, y_pred_sgd, alpha=0.5, color="orange")   # Scatter plot: Actual vs Predicted (SGD Regressor)
axes[2, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Ideal line
axes[2, 0].set_title("SGD Regressor: Predicted vs Actual")            # Set title
axes[2, 0].set_xlabel("Actual Composite Success (Scaled)")           # Set x-axis label
axes[2, 0].set_ylabel("Predicted Composite Success (Scaled)")        # Set y-axis label

axes[2, 1].hist(residuals_sgd, bins=30, edgecolor='k', alpha=0.7, color="orange")  # Histogram of residuals (SGD Regressor)
axes[2, 1].set_title("SGD Regressor: Residual Distribution")         # Set title
axes[2, 1].set_xlabel("Residuals")                                   # Set x-axis label
axes[2, 1].set_ylabel("Frequency")                                   # Set y-axis label

plt.tight_layout()     # Adjust layout so subplots do not overlap
plt.show()             # Display the combined figure

print("Script completed.")  # Print completion message

import joblib

# Na het trainen van je modellen
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(sgd_model, 'sgd_model.pkl')
import json

# Na het berekenen van de target_mapping
with open('target_mapping.json', 'w') as f:
    json.dump(target_mapping, f)