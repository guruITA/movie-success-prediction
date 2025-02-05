import pandas as pd 

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/IMDb movies.csv") 

# # Show first 5 rows of the dataframe (info from dataset)
# print("<------------------------>")
# print("With head(): ",df.head(5))
# print("<------------------------>")

# # Show last 5 rows of the dataframe (info from dataset)
# print("<------------------------>")
# print("With tail(): ",df.tail(5))
# print("<------------------------>")

# Show info about the dataset (column names, data type, amount of rows or columns, etc. )
print("<------------------------>")
print("With info(): ")
df.info()
print("<------------------------>")

# # Show statical summary of nummerical  columns (count, mean, etc.)
# print("<------------------------>")
# print("With describe(): ",df.describe())
# print("<------------------------>")



# remove Nan values from budget
df1 = df.dropna(subset=["budget"])

# Get only budget fields that contain $ 
df1 = df1[df1["budget"].str.contains("$", regex=False)]

# remove $ symbol from de fields and put them back in budget field to save changes
df1["budget"] = df1["budget"].str.replace("$", "", regex=False)

# change financial objects to float
df1["budget"] = df1["budget"].astype(float)

# print("test change finacial data types",df1["budget", "usa_gross_income", "worldwide_gross_income"])

# get total budget per year
test = df1.groupby("year").agg(
    budget_per_year=("budget", "sum")  
).reset_index()

# show first 5 rows
print(test.head(20))

# morgen 
print("genre", df1["genre"])

# pd.factorize()

# Graphic with pyplot
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# plt.title("Budget per year")
# plt.bar(test["year"], test["budget_per_year"], color="red", label="Budget per year")
# plt.xlabel("Year")
# plt.ylabel("Budget in $")
# plt.yscale("log")
# formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')  
# plt.gca().yaxis.set_major_formatter(formatter)  

# plt.legend()
# plt.show()