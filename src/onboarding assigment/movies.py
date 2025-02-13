import pandas as pd 

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/IMDb movies.csv")

# Show first 5 rows of the dataframe (info from dataset)
print("<------------------------>")
print("With head(): ",df.head(5))
print("<------------------------>")

# Show last 5 rows of the dataframe (info from dataset)
print("<------------------------>")
print("With tail(): ",df.tail(5))
print("<------------------------>")

# Show info about the dataset (column names, data type, amount of rows or columns, etc. )
print("<------------------------>")
print("With info(): ")
df.info()
print("<------------------------>")

# Show statical summary of nummerical  columns (count, mean, etc.)
print("<------------------------>")
print("With describe(): ",df.describe())
print("<------------------------>")

# Making sure to make a copy of dataframe
df1 = df.copy()

# Removes NaN value from budget column 
df1 = df1.dropna(subset=["budget"])

# Get only budget fields that contain $ and saving in another dataframe
dollars = df1[df1["budget"].str.contains("$", regex=False)].copy()

# Remove $ symbol from de fields and put them back in budget field to save changes
dollars["budget"] = dollars["budget"].str.replace("$", "", regex=False)

# Change budget columns type to float
dollars["budget"] = dollars["budget"].astype(float)

# Checking if budget is changed to float
assert dollars["budget"].dtype == float

# Checking types of each column
print(dollars.dtypes)

# Get total budget per year
get_budget_per_year = dollars.groupby("year").agg(
    budget_per_year=("budget", "sum")  
).reset_index()

# Show first 5 rows
print(get_budget_per_year.head(20))

# Creating list of columns
finacial = ["usa_gross_income"]

# Loop thru each column in the list 
for col in finacial:
    # Remove everything that is not a number 
    df[col] = df[col].str.replace(r"[^\d.]", "", regex=True)

    # Convert to a numeric type (int or float) output is float
    df[col] = pd.to_numeric(df[col])

categorical_col = ['genre', 'country', 'language', 'director', 'writer', 'production_company']

# Store original labels in separate columns (optional)
for col in categorical_col:
    # Factorize the column
    encoded_values = pd.factorize(df[col].fillna("Unknown"))[0]
    
    # Create a new column to store the encoded labels for use in the model
    df[f"{col}_encoded"] = encoded_values

# Printing types of all columns 
print(df.dtypes)

# Splitting genres of each movie in lists
df["genre"] = df["genre"].str.split(", ")

# Getting film title with their genres
print("test\n", df[["title", "genre"]].head(5))

# First, the 'genre' column is exploded, meaning that lists of genres are split into separate rows, 
# each containing a single genre while retaining the original index. 
# Then, these genres are converted into binary columns (one-hot encoding).
# Finally, the rows are grouped back by the original movie index, summing up the binary values 
# to ensure each movie has a single row with its corresponding genres encoded.
genre_dummies = df["genre"].explode().str.get_dummies().groupby(level=0).sum()

# Adding the data to the original dataframe 
df = pd.concat([df, genre_dummies], axis=1)

# Printing the first 5 rows with the one hot encodeing
print("Hot encoding\n", df.head(5))

# Printing the first 10 rows of genre
print("First 10 rows of genre\n", df["genre"].head(10))

# Split each genre to seperate rows and then count then amount of times each genre appears in dataset
genre_counts = df["genre"].explode().value_counts()
print("Value counts\n", genre_counts)

# Making a array list with all unique genres
unique_genres = df["genre"].explode().unique()

# Convert categorical data to numerical data
code = pd.factorize(unique_genres)[0]

# Getting unique genre and numerical data in seperate lists
print("test1\n", [unique_genres, str(code)])    

# Creating an empty list to store genres with their numeric code
genre_code_list = []

# The zip() function takes two lists (here 'unique_genres' and 'code') and creates pairs of elements.
# It combines the first element of the first list with the first element of the second list,
# the second element of the first list with the second element of the second list, and so on.
# For example, zip(["Drama", "Comedy"], [1, 2]) would create the pairs ("Drama", 1) and ("Comedy", 2).

# Loop through each paired value (genre, code) created by zip() 
for genre, code in zip(unique_genres, code):

    # Adding data to genre_code_list in a list each time it iterates
    genre_code_list.append([genre, str(code)])

    # Print genre and code each iteration
    print(f"genre: {genre}: {code}")

# Print the filled list with genre and code 
print(genre_code_list)

# Graphic with pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Title of the graphic 
plt.title("Budget per year")

# Addig x-axis value and y-axis value, color of bar that shows result is red and adding label so that it can be shown in legend later
plt.bar(get_budget_per_year["year"], get_budget_per_year["budget_per_year"], color="red", label="Budget per year ($)")   

plt.xlabel("Year")
plt.ylabel("Budget in $")

# The lambda function takes a number 'x' and formats it to include commas as thousands separators and no decimal places.
formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}') 

# Apply the custom formatter to the y-axis of the current plot
plt.gca().yaxis.set_major_formatter(formatter)  

# To show legend
plt.legend()

# To show grafich
plt.show()

# Getting Info of the dataframe 
df.info()

# Only gets the numbers from the budget en put in new column value
df["value"] = df["budget"].str.extract(r'(\d+)')
df["worlwide_gross_income"] = df["worlwide_gross_income"].str.extract(r'(\d+)')

# Gets de text or symbols from the budget en put in new column currency
df["currency"] = df["budget"].str.extract(r'([^\d\s]+)')

# Convert value to numberic
df["value"] = pd.to_numeric(df["value"])
df["worlwide_gross_income"] = pd.to_numeric(df["worlwide_gross_income"])

# Sort year in ASC
df = df.sort_values("year", ascending=True)

# Making a new dataframe with the columns I need
new_df = df.copy()

# Getting unique currencies so not the same currencies the same time
currecies = new_df["currency"].unique()

# Getting the amount of the currencies and what kind of currencies there are
print(len(currecies), currecies)

# Function to convert currecies to dollars
def convert_to_usd(value, currency):
    match currency:
        case "$": return value * 1.0        # USD stays the same
        case "ROL": return value * 0.20742  # Romanian Leu
        case "ITL": return value * 0.00054  # Italian Lira
        case "SEK": return value * 0.096    # Swedish Krona
        case "FRF": return value * 0.153    # French Franc
        case "NOK": return value * 0.096    # Norwegian Krone
        case "GBP": return value * 1.27     # British Pound
        case "DEM": return value * 0.568    # German Mark
        case "PTE": return value * 0.006    # Portuguese Escudo
        case "CZK": return value * 0.044    # Czech Koruna
        case "INR": return value * 0.012    # Indian Rupee
        case "CAD": return value * 0.79     # Canadian Dollar
        case "CHF": return value * 1.09     # Swiss Franc
        case "FIM": return value * 0.168    # Finnish Markka
        case "JPY": return value * 0.0072   # Japanese Yen
        case "BEF": return value * 0.025    # Belgian Franc
        case "NLG": return value * 0.51     # Dutch Guilder
        case "DKK": return value * 0.14     # Danish Krone
        case "KRW": return value * 0.00076  # South Korean Won
        case "PLN": return value * 0.26     # Polish Zloty
        case "RUR": return value * 0.011    # Russian Ruble (Old)
        case "AUD": return value * 0.66     # Australian Dollar
        case "TRL": return value * 0.000051 # Turkish Lira (Old)
        case "NZD": return value * 0.62     # New Zealand Dollar
        case "HKD": return value * 0.13     # Hong Kong Dollar
        case "EGP": return value * 0.032    # Egyptian Pound
        case "ESP": return value * 0.006    # Spanish Peseta
        case "CNY": return value * 0.14     # Chinese Yuan
        case "EUR": return value * 1.08     # Euro
        case "ALL": return value * 0.0096   # Albanian Lek
        case "ISK": return value * 0.0072   # Icelandic Króna
        case "DOP": return value * 0.017    # Dominican Peso
        case "BDT": return value * 0.0092   # Bangladeshi Taka
        case "IEP": return value * 1.27     # Irish Pound
        case "IRR": return value * 0.000024 # Iranian Rial
        case "HRK": return value * 0.14     # Croatian Kuna
        case "JMD": return value * 0.0065   # Jamaican Dollar
        case "SIT": return value * 0.0045   # Slovenian Tolar
        case "PHP": return value * 0.018    # Philippine Peso
        case "SGD": return value * 0.73     # Singapore Dollar
        case "BRL": return value * 0.20     # Brazilian Real
        case "HUF": return value * 0.0029   # Hungarian Forint
        case "EEK": return value * 0.068    # Estonian Kroon
        case "BND": return value * 0.73     # Brunei Dollar
        case "THB": return value * 0.029    # Thai Baht
        case "GRD": return value * 0.003    # Greek Drachma
        case "ZAR": return value * 0.053    # South African Rand
        case "ATS": return value * 0.078    # Austrian Schilling
        case "IDR": return value * 0.000066 # Indonesian Rupiah
        case "PKR": return value * 0.0036   # Pakistani Rupee
        case "CLP": return value * 0.0012   # Chilean Peso
        case "BGL": return value * 0.56     # Bulgarian Lev
        case "NPR": return value * 0.0075   # Nepalese Rupee
        case "MXN": return value * 0.059    # Mexican Peso
        case "YUM": return value * 0.0096   # Yugoslav Dinar
        case "LTL": return value * 0.34     # Lithuanian Litas
        case "MYR": return value * 0.21     # Malaysian Ringgit
        case "PYG": return value * 0.00014  # Paraguayan Guarani
        case "SKK": return value * 0.034    # Slovak Koruna
        case "MTL": return value * 2.58     # Maltese Lira
        case "TWD": return value * 0.031    # Taiwan Dollar
        case "UAH": return value * 0.027    # Ukrainian Hryvnia
        case "ARS": return value * 0.0023   # Argentine Peso
        case "COP": return value * 0.00025  # Colombian Peso
        case "LVL": return value * 1.62     # Latvian Lats
        case "VND": return value * 0.000041 # Vietnamese Dong
        case "NGN": return value * 0.0013   # Nigerian Naira
        case "VEB": return value * 0.00010  # Venezuelan Bolivar
        case "ILS": return value * 0.27     # Israeli Shekel
        case "TTD": return value * 0.15     # Trinidad and Tobago Dollar
        case "GEL": return value * 0.38     # Georgian Lari
        case "AZM": return value * 0.00012  # Azerbaijani Manat
        case "RON": return value * 0.21     # Romanian Leu (New)
        case "LKR": return value * 0.0032   # Sri Lankan Rupee
        case "MNT": return value * 0.00029  # Mongolian Tögrög
        case "AMD": return value * 0.0026   # Armenian Dram
        case "AED": return value * 0.27     # United Arab Emirates Dirham
        case _: return value                # Unknown currency, keep value unchanged

# Convert values of different currencies to USD
new_df["budget_usd"] = new_df.apply(lambda row: convert_to_usd(row["value"], row["currency"]), axis=1)

# Creating converted_currency column from converted_currency
new_df["converted_currency"] = new_df["currency"]

# Put USD as value in all column
new_df["converted_currency"] = "USD"

# If column value is NaN it will be changed to 0
new_df["value"] = new_df["value"].fillna(0)
new_df["budget_usd"] = new_df["budget_usd"].fillna(0)
new_df["budget"] = new_df["budget"].fillna(0)
new_df["worlwide_gross_income"] = new_df["worlwide_gross_income"].fillna(0)
new_df["country"] = new_df["country"].fillna("Unknown")

# If column value is NaN it will be changed to Unknown
new_df["currency"] = new_df["currency"].fillna("Unknown")

# If worlwide_gross_income is greater then value_usd output will be 1 if not it will be 0
new_df["successful"] = new_df.apply(lambda row: 1 if row["worlwide_gross_income"] > row["budget_usd"] else 0, axis = 1)

# Printing the first 20 rows 
print(new_df.head(20))

# Converting dataframe info to csv file
new_df.to_csv("data/clean_movies_successfull.csv")

new_df.info()