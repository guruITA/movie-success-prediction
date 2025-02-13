import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
# import logging

load_dotenv()

# Connection details
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# Format the connection URL for MySQL using pymysql
connection_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(connection_url)

try:
    # Loading the dataset
    csv_file = "data/clean_movies_successfull.csv"

    # If table excist, replace it 
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Error: The file '{csv_file}' was not found.")
    
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded dataset with {len(df)} records.")

    try:
        # Create a connection to the database
        with engine.connect() as connection:
            print("Connecting to the database...")

            # Verify database connection
            stmt = text("SELECT VERSION()")
            result = connection.execute(stmt).fetchone()
            print(f"Connected to MySQL successfully! MySQL version: {result[0]}")

            # Insert data in chunks for large datasets
            df.to_sql("movies", engine, if_exists="replace", index=False, chunksize=500)
            print("IMDb movies dataset stored in the database successfully!")

    except SQLAlchemyError as e:
        print(f"Error during data insertion: {e}")

except Exception as e:
    print("Error connecting to the database:", e)