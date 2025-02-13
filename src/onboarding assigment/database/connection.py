import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# Connection details
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# Format the connection URL for MySQL
connection_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

# Create SQLAlchemy engine
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
            print("IMDb movies dataset stored in the database successfully")

            # Function to execute SQL queries
            def run_query(query):
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(query))
                        return result.fetchall()
                except SQLAlchemyError as e:
                    print(f"Error executing query: {e}")

            # Get all movies with an English language
            try:
                query_english_movies = "SELECT * FROM movies WHERE language = '%English%'"
                english_movies = run_query(query_english_movies)
                print(f"Found {len(english_movies)} movies with English language")
            except SQLAlchemyError as e:
                print(f"Error fetching movies with english language: {e}")

            # Get the number of movies for each country
            try:
                query_movies_per_countries = "SELECT title, country FROM movies"
                movies_per_countries = run_query(query_movies_per_countries)

                df_movies = pd.DataFrame(movies_per_countries)
                df_movies["country"] = df_movies["country"].str.split(", ")
                country_counts = df_movies["country"].explode().value_counts()
                pd.set_option("display.max_rows", None)
                print("Amount of movies in each country\n", country_counts)
            except SQLAlchemyError as e:
                print(f"Error fetching movies: {e}")
            
            # Get the average movie budget for each year
            try:
                query_avg_budget_per_year = """
                    SELECT year, AVG(budget_usd) AS avg_budget
                    FROM movies
                    GROUP BY year
                    ORDER BY year ASC;
                """
                avg_budget_per_year = run_query(query_avg_budget_per_year)
                print("Average budget per year in dollars:\n", avg_budget_per_year)
                
            except SQLAlchemyError as e:
                print(f"Error fetching average budget per year: {e}")

    except SQLAlchemyError as e:
        print(f"Error during data insertion: {e}")

except Exception as e:
    print(f"Error connecting to the database: {e}")
