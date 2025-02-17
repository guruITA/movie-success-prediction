import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Gets environment variables from the .env file
load_dotenv()

# Gets de connection details from de env file (variables)
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# Format the connection URL for MySQL
connection_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

# Create engine for database connection 
engine = create_engine(connection_url)

try:
    # Path for csv file 
    csv_file = "data/clean_movies_successfull.csv"

    # Checking if the csv file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Error: The file '{csv_file}' was not found.")
    
    # Loading the data from csv file into dataFrame
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded dataset with {len(df)} records.")

    try:
        # Creating a connection to the database
        with engine.connect() as conn:
            print("Connecting to the database...")

            # Checking connething by de database version 
            stmt = text("SELECT VERSION()")

            # Execute the SQL statement and fetch the first result
            result = conn.execute(stmt).fetchone()

            # Print a confirmation message with MySQL version
            print(f"Connected to MySQL successfully! MySQL version: {result[0]}")

            # Insert data into database if table already exists it will be replaced with new data 
            df.to_sql("movies", engine, if_exists="replace", index=False, chunksize=500)
            print("IMDb movies dataset stored in the database successfully")

            # Function to execute SQL queries
            def run_query(query):
                try:
                    # Creating a connection to the database
                    with engine.connect() as conn:

                        # Execute the query fetch all results
                        result = conn.execute(text(query))
                        # Fetch all results
                        return result.fetchall()
                    
                except SQLAlchemyError as e:
                    # Print an error message if the query execution fails
                    print(f"Error executing query: {e}")

            try:
                # Get all movies with an English language
                query_english_movies = "SELECT * FROM movies WHERE language LIKE '%English%'"

                # Executing the query using the run_query function 
                english_movies = run_query(query_english_movies)

                # Printing the amount of English language movies
                print(f"Found {len(english_movies)} movies with English language")
            except SQLAlchemyError as e:
                # Print an error message if the query execution fails
                print(f"Error fetching movies with english language: {e}")


            try:
                # Get the number of movies for each country
                query_movies_per_countries = "SELECT title, country FROM movies"

                # Executing the query using the run_query
                movies_per_countries = run_query(query_movies_per_countries)

                # Converting the results into a dataFrame
                df_movies = pd.DataFrame(movies_per_countries)

                # Splitting country of each movie in lists
                df_movies["country"] = df_movies["country"].str.split(", ")

                # The country column is exploded, meaning that lists of genres are split into separate rows,
                # each containing a single country while retaining the original index. 
                # And then count then amount of times each movie appears in a country
                country_counts = df_movies["country"].explode().value_counts()

                # With this I get to see all rows in the output 
                pd.set_option("display.max_rows", None)

                # Printing the number of movies per country
                print("Amount of movies in each country\n", country_counts)
            except SQLAlchemyError as e:
                # Print an error message if the query execution fails
                print(f"Error fetching movies: {e}")
            
            try:
                # Get the average movie budget for each year
                query_avg_budget_per_year = """
                    SELECT year, AVG(budget_usd) AS avg_budget
                    FROM movies
                    GROUP BY year
                    ORDER BY year ASC;
                """

                # Executing the query using the run_query
                avg_budget_per_year = run_query(query_avg_budget_per_year)

                # Print the average movie budget per year
                print("Average budget per year in dollars:\n", avg_budget_per_year)
                
            except SQLAlchemyError as e:
                # Print an error message if the query execution fails
                print(f"Error fetching average budget per year: {e}")

    except SQLAlchemyError as e:
        # Print an error message if there is an issue inserting data into the database
        print(f"Error during data insertion: {e}")

except Exception as e:
    # Printing an error message if there is an issue connecting to the database
    print(f"Error connecting to the database: {e}")
