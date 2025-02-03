import pandas as pd
# more ugly df= pd.read_csv('IMDb movies.csv')
df= pd.read_csv('data/IMDb movies.csv',sep=',')

df.head(5)
df.shape
df.dtypes
df.describe()  # remember math?

# always make a copy
df1=df

df1['budget'].head(5)

# DROP ALL EMPTY ROWS
df1 = df.dropna(subset=['budget'])
df1['budget'].head(5)

# CHECK HOW regex WORKS
df1['budget'].head(5).str.contains('$',regex=False)

# SELECT ROWS WITH 'bugdet' CONTAINS "$" , PUT INTO NEW DF
dfUSA = df1.loc[lambda x: x['budget'].str.contains('$',regex=False) ]

dfUSA[['budget']].head(5)

# CHECK TYPES
dfUSA.dtypes

# REPLACE '$' WITH NOTHING, TO MAKE A CONVERSION POSSIBLE
dfUSA['budget'] = dfUSA['budget'].str.replace('$','')


dfUSA[['title','budget']].head()

# CONVERT TO NUMBER
dfUSA['budget']=pd.to_numeric(dfUSA['budget'])
dfUSA.dtypes

# NOW READY TO DO A SUM FOR EACH YEAR
# result is a serie, CAN NOT EASILY BE USED FOR PLOTTING
serieUSAGrouped = dfUSA.groupby('year')['budget'].sum()

# result is a dataframe< CAN BE USED FOR PLOTTING

dfUSAGrouped = (
                    dfUSA
                    .groupby('year')
                    .agg({'budget':'sum'})
                )



import plotly.express as px

fig = px.bar(dfUSAGrouped,
            x=dfUSAGrouped.index
            ,y='budget')
fig.show()

# EXERCISE 2, ONLY USA DATA
# USE get_dummies  , DOES THE WHOLE JOB FOR YOU
dfUSA['genre'].head(5)

dfUSA['genre'].head(5).str.get_dummies(sep=', ')


#  CONCAT AXIS=1, GLUE COLUMNS
dfUSASplitGenre=pd.concat([dfUSA,dfUSA['genre'].str.get_dummies(sep=', ')],axis=1)
dfUSASplitGenre.head(5)


# 1 COMMA ==> 2 actors
dfUSA['numberOfActors']=dfUSA['actors'].str.count(',')+1
# See error: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

# Or First a Serie
dfUSA = df1.loc[lambda df1: df1['budget'].str.contains('$',regex=False) ]

serieUSANumberOfActors=dfUSA['actors'].str.count(',')+1

dfNumberOfActors=pd.DataFrame(serieUSANumberOfActors )
dfNumberOfActors=dfNumberOfActors.rename(columns = {'actors':'numberOfActors'})

dfUSA=pd.concat([dfUSA,dfNumberOfActors],axis=1)


dfUSA[['numberOfActors']].head(5)

#SELECT MOVIES WITH ONLY 1 ACTOR

dfUSA[['actors','numberOfActors']].loc[lambda df: df['numberOfActors']==1].head(5)

# EXTRA COLUMN SPLIT, TAKE FIRST
dfUSA['mainActor']=dfUSA['actors'].str.split(',').str[0]

dfUSA[['actors','numberOfActors','mainActor']].loc[lambda df: df['numberOfActors']==3].head(5)