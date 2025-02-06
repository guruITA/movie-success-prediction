import pandas as pd 

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/IMDb movies.csv")

# Replace NaN values to empty strings
df["budget"] = df["budget"].fillna("").astype(str)
df["worlwide_gross_income"] = df["worlwide_gross_income"].fillna("").astype(str)
df["usa_gross_income"] = df["usa_gross_income"].fillna("").astype(str)

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
columns = ["budget", "usa_gross_income", "worlwide_gross_income"]

# Loop thru each column in the list 
for col in columns:
    # Remove everything that is not a number 
    df[col] = df[col].str.replace(r"[^\d.]", "", regex=True)

    # Convert to a numeric type (int or float) output is float
    df[col] = pd.to_numeric(df[col])

# Printing types of all columns 
print(df.dtypes)

# Graphic with pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.title("Budget per year")
plt.bar(get_budget_per_year["year"], get_budget_per_year["budget_per_year"], color="red", label="Budget per year")
plt.xlabel("Year")
plt.ylabel("Budget in $")
plt.yscale("log")
formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')  
plt.gca().yaxis.set_major_formatter(formatter)  

plt.legend()
plt.show()