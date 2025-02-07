import pandas as pd

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/netflix_data.csv")

# Show info about the dataset (column names, data type, amount of rows or columns, etc. )
df.info()

# Remove all rows where duration column is NaN 
df1 = df.dropna(subset=["duration"])

# Getting data that only contains movies 
df1 = df1[df1["type"].str.contains("Movie", regex=False)]

# Getting data that is between 1990 and 2000
df1 = df1[(df1["release_year"] > 1990) & (df1["release_year"] < 2000)]

# Getting data that only contains min in it 
df1 = df1[df1["duration"].str.contains("min", regex=False)]

# Removing min from the field
df1["duration_minutes"] = df1["duration"].str.replace(" min", "", regex=False).astype(int)

# Getting the most common time 
print(df1["duration_minutes"].mode()[0])

# If duration of moveis is under 90 minutes put 1 or 0 in short_movies column
df1["short_movies"] = df1.apply(lambda row: 1 if row["duration_minutes"] < 90 else 0, axis = 1)

# Getting actions movies
df_action = df1[df1["listed_in"].str.contains("Action")]

# Counting the amount of short action movies
print(df_action["short_movies"].sum())

# Replacing the type column type using hot encoding
df[['type_movie', 'type_tvshow']] = df["type"].str.get_dummies()

# Printing dataset
print(df.head(5))