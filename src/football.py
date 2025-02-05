import pandas as pd

# Get data from de csv file 
df = pd.read_csv("data/football_results.csv")

# Printing the first rows of de dataframe (Data from csv)
print(df.head())

# Getting info of the dataframe
df.info()

# Printing de unique number of home teams 
print("Home teams: ", len(df.home.unique()))

# Printing de away number of home teams 
print("Away teams: ", len(df.away.unique()))

difference = len(df.away.unique()) - len(df.home.unique())

print("Difference " , difference)

# Putting data into set 
home = set(df.home)
away = set(df.away)

z = home.difference(away)
a = away.difference(home)

print("Difference away team - home team" , len(z))
print("Difference home team - away team" , len(a))

df["home_wins"] = df.apply(lambda row: 1 if row.gh > row.ga else 0, axis = 1)
df["home_draws"] = df.apply(lambda row: 1 if row.gh == row.ga else 0, axis = 1)
df["home_loses"] = df.apply(lambda row: 1 if row.gh < row.ga else 0, axis = 1)

print(df.head())

test = df.groupby("home").agg(
    wins = ("home_wins", "sum"),
    games_played = ("home_wins", "count")
)

sorted_df = test.sort_values("wins", ascending=False)

print("sorted table", sorted_df.head(10))

test1 = df.groupby("home").agg(
    wins=("home_wins", "sum"),
    draws=("home_draws", "sum"),
    loses=("home_loses", "sum"),
    games_played=("home_wins", "count") 
)

test1["perc_wins"] = test1.apply(lambda row: row["wins"] / row["games_played"], axis=1)
test1["perc_draws"] = test1.apply(lambda row: row["draws"] / row["games_played"], axis=1)
test1["perc_loses"] = test1.apply(lambda row: row["loses"] / row["games_played"], axis=1)

filtered_teams = test1[test1.apply(lambda row: row["games_played"] >= 500, axis=1)]
sorted_teams = filtered_teams.sort_values("perc_wins", ascending=False)

print(sorted_teams.head(10))

import matplotlib.pyplot as plt

plt.bar(sorted_teams.index[:10], sorted_teams["perc_wins"].head(10),
        color="blue", 
        label="perc_home_wins")

plt.bar(sorted_teams.index[:10], sorted_teams["perc_draws"].head(10),
        color="red", 
        label="perc_home_draws",
        bottom=sorted_teams["perc_wins"].head(10)) 

plt.bar(sorted_teams.index[:10], sorted_teams["perc_loses"].head(10),
        color="yellow", 
        label="perc_home_loses",
        bottom=sorted_teams["perc_wins"].head(10) + sorted_teams["perc_draws"].head(10))  

plt.xticks(rotation=90)
plt.title("Home")
plt.ylabel("Percentage")
plt.xlabel("Home")
plt.legend(title = "Metric")
plt.show()