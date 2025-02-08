import pandas as pd

# Read the dataset (csv file) into dataframe
df = pd.read_csv("data/crimes.csv")

# Making sure to make a copy of dataframe
df1 = df

# Show info about the dataset (column names, data type, amount of rows or columns, etc. )
df1.info()

# Checking types of each column
print(df1.dtypes)

# Show first 10 rows of the dataframe (info from dataset)
print(df1.head(10))

df1["TIME OCC"] = df1["TIME OCC"].astype(str).str.zfill(4)
df1["TIME OCC"] = pd.to_datetime(df1["TIME OCC"], format="%H%M").dt.hour

# Creating peak_crimes_hour column with the data from TIME OCC
df1["peak_crime_hour"] = df1["TIME OCC"]

# Printing the hour where the most crimes happen
print("Common hour crimes appear: ", df1["peak_crime_hour"].mode()[0])

# Filter for crimes that happened between 22pm and 03:59am
crimes_between_22_4 = df1[(df1["TIME OCC"] >= 22) | (df1["TIME OCC"] <= 4)]

# Printing the crimes that happend between 22 pm and 3:59am
print(crimes_between_22_4[["TIME OCC", "AREA NAME"]].head(10))

# Getting the amount of crimes per area
crime_counts_per_area = crimes_between_22_4["AREA NAME"].value_counts()

# Printing the amount of crimes per area
print("Counted up crimes per area:\n", crime_counts_per_area)

# Getting the area with the highest amount of crimes between 22pm and 3:59am
crime_counts_area = crime_counts_per_area.idxmax()

# Getting the highest amount of crimes between 22pm and 3:59am
highest_crime_count = crime_counts_per_area.max()

# Printing the area with highest crimes between 22pm and 3:59am
print(f"Area {crime_counts_area} has the largets frequency of crime counted up to {highest_crime_count} crimes between 22pm and 3:59pm")

# 
age_group_labels = ["0-17", "18-25", "26-34", "35-44", "45-54", "55-64", "65+"]

# 
df["victim_age_group"] = pd.qcut(df["Vict Age"], q = len(age_group_labels), labels=age_group_labels)

# 
print("test\n", df["victim_age_group"].head(100))

# 
print(df["victim_age_group"].value_counts())

# 
crime_count_per_age_group = df["victim_age_group"].value_counts()
print("get index\n", crime_count_per_age_group.index, "values " , crime_count_per_age_group.values)

# 
columns = [
            "Age group",
            "amount crime"
        ]

# 
data = list(zip(    
            crime_count_per_age_group.index,
            crime_count_per_age_group.values
            ))

# 
created_df = pd.DataFrame(data, columns=columns)

# 
print("created dataframe\n", created_df)