import pandas as pd 

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/IMDb movies.csv")

# Replace NaN values to empty strings
df["budget"] = df["budget"].fillna("")
df["worlwide_gross_income"] = df["worlwide_gross_income"].fillna("")
df["usa_gross_income"] = df["usa_gross_income"].fillna("")

# Print output of budget to check if NaN input of column is changed to empty string
print(df[["budget"]].head(10))

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
df1 = df

# removes NaN value from budget column 
# df1 = df1.dropna(subset=["budget"])

# Get only budget fields that contain $ and saving in another dataframe
dollars = df1[df1["budget"].str.contains("$", regex=False)]

# Remove $ symbol from de fields and put them back in budget field to save changes
dollars["budget"] = dollars["budget"].str.replace("$", "", regex=False)

# Change budget columns type to float
dollars["budget"] = dollars["budget"].astype('float')

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

# Making copy of dataframe to change the types of some finacial columns
df2 = df

# Creating list of columns
finacial = ["budget", "usa_gross_income", "worlwide_gross_income"]

# Loop thru each column in the list 
for col in finacial:
    # Remove everything that is not a number 
    df2[col] = df2[col].str.replace(r"[^\d.]", "", regex=True)

    # Convert to a numeric type (int or float) output is float
    df2[col] = pd.to_numeric(df[col])

# Creating list of columns
categorical = ["country", "language", "director", "writer", "production_company"]

# Loop thru each column in the list 
for col in categorical:

    # Convert categorical data to numerical data
    df1[col] = pd.factorize(df1[col])[0]

# Printing types of all columns 
print(df2.dtypes)

# Splitting genres of each film in lists
df2["genre"] = df2["genre"].str.split(", ")

# Getting film title with their genres
print("test\n", df2[["title", "genre"]].head(5))

genre_dummies = df2["genre"].explode().str.get_dummies().groupby(level=0).sum()

test = pd.concat([df2["title"], genre_dummies], axis=1)

print("hot encoding\n", test.head(5))

print("test0\n", df2["genre"].head(10))

# Split each genre to seperate rows and then count then amount of times each genre appears in dataset
genre_counts = df2["genre"].explode().value_counts()
print("Value counts\n", genre_counts)

# Making a array list with all unique genres
unique_genres = df2["genre"].explode().unique()

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