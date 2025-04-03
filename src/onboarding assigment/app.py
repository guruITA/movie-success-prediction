import dash
from dash import dcc, html, Input, Output
from dash import dash_table
import plotly.express as px
import pandas as pd

file_path = 'data/clean_movies_successfull.csv'
movies_df = pd.read_csv(file_path)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard", style={'textAlign': 'center', 'padding': '20px'}),

    html.Div([ 
        html.Div([
            html.H3("Filters", style={'margin-bottom': '20px'}),

            dcc.Dropdown(
                id='title-filter',
                options=[{'label': title, 'value': title} for title in movies_df['title'].unique()],
                multi=True,
                placeholder="Selecteer film titel",
                style={'margin-bottom': '20px'}
            ),

            dcc.Dropdown(
                id='genre-filter',
                options=[{'label': genre, 'value': genre} for genre in movies_df['genre'].str.split(',').explode().unique()],
                multi=True,
                placeholder="Selecteer genres",
                style={'margin-bottom': '20px'}
            ),

            html.Div([
                html.Label("Selecteer Jaarperiode", style={'font-size': '16px'}),
                dcc.DatePickerRange(
                    id='year-picker',
                    start_date=str(movies_df['year'].min()),
                    end_date=str(movies_df['year'].max()),
                    display_format='YYYY',
                    start_date_placeholder_text="Startdatum",
                    end_date_placeholder_text="Einddatum",
                    style={'margin-bottom': '20px'}
                ),
            ]),

            dcc.Dropdown(
                id='success-filter',
                options=[
                    {'label': 'Succesvol', 'value': 1},
                    {'label': 'Niet Succesvol', 'value': 0}
                ],
                multi=True,
                placeholder="Selecteer successtatus",
                style={'margin-bottom': '20px'}
            ),

            dcc.Dropdown(
                id='language-filter',
                options=[{'label': lang, 'value': lang} for lang in movies_df['language'].unique()],
                multi=True,
                placeholder="Selecteer taal",
                style={'margin-bottom': '20px'}
            ),

            dcc.Dropdown(
                id='actor-filter',
                options=[{'label': actor, 'value': actor} for actor in movies_df['actors'].str.split(',').explode().unique()],
                multi=True,
                placeholder="Selecteer acteur",
                style={'margin-bottom': '20px'}
            ),

            dcc.Dropdown(
                id='director-filter',
                options=[{'label': director, 'value': director} for director in movies_df['director'].unique()],
                multi=True,
                placeholder="Selecteer directeur",
                style={'margin-bottom': '20px'}
            ),

        ], style={'width': '30%', 'padding': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            dcc.Graph(id='genre-chart'),
            dcc.Graph(id='language-chart'),
            dcc.Graph(id='avg-budget-year-chart'),
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),

    ], style={'display': 'flex', 'padding': '20px'}),

    html.Div([ 
        html.H3("Tabel met Films", style={'textAlign': 'center', 'padding': '20px'}),
        dash_table.DataTable(
            id='movies-table',
            columns=[
                {"name": "Title", "id": "title"},
                {"name": "Year", "id": "year"},
                {"name": "Director", "id": "director"},
                {"name": "Actor", "id": "actors"},
                {"name": "Budget", "id": "budget_usd"},
                {"name": "Average Vote", "id": "avg_vote"}
            ],
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            page_size=10,  
        )
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '10px', 'margin-top': '40px'})
])

@app.callback(
    [Output('genre-chart', 'figure'),
     Output('language-chart', 'figure'),
     Output('avg-budget-year-chart', 'figure'),
     Output('movies-table', 'data')],
    [Input('title-filter', 'value'),  
     Input('genre-filter', 'value'),
     Input('year-picker', 'start_date'),
     Input('year-picker', 'end_date'),
     Input('success-filter', 'value'),
     Input('language-filter', 'value'),
     Input('actor-filter', 'value'),
     Input('director-filter', 'value')]
)
def update_graphs(selected_titles, selected_genres, start_date, end_date, selected_success, selected_languages, selected_actors, selected_directors):
    filtered_df = movies_df.copy()

    if selected_titles:
        filtered_df = filtered_df[filtered_df['title'].isin(selected_titles)]

    if selected_genres:
        selected_genres_set = set([genre.lower().strip() for genre in selected_genres])

        filtered_df = filtered_df[filtered_df['genre'].str.split(',').apply(
            lambda genres: any(genre.strip().lower() in selected_genres_set for genre in genres))]


    if selected_languages:
        selected_languages_set = set([language.lower().strip() for language in selected_languages])
        filtered_df = filtered_df[filtered_df['language'].apply(
            lambda x: any(language.strip().lower() in selected_languages_set for language in x.split(',')))]

    if selected_actors:
        selected_actors_set = set([actor.lower().strip() for actor in selected_actors])
        filtered_df = filtered_df[filtered_df['actors'].apply(
            lambda x: any(actor.strip().lower() in selected_actors_set for actor in x.split(',')))]

    if selected_directors:
        selected_directors_set = set([director.lower().strip() for director in selected_directors])
        filtered_df = filtered_df[filtered_df['director'].apply(
            lambda x: any(director.strip().lower() in selected_directors_set for director in x.split(',')))]

    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    filtered_df = filtered_df[(filtered_df['year'] >= start_year) & (filtered_df['year'] <= end_year)]

    if selected_success:
        filtered_df = filtered_df[filtered_df['successful'].isin(selected_success)]

    genres = filtered_df['genre'].str.split(',').explode().str.strip().str.lower()
    genre_count = genres.value_counts()
    genre_df = genre_count.reset_index()
    genre_df.columns = ['Genre', 'Aantal films']

    genre_title = f"Genres"
    genre_fig = px.bar(genre_df, 
                       x='Genre', 
                       y='Aantal films', 
                       labels={'x': 'Genre', 'y': 'Aantal films'},
                       title=genre_title)
    
    genre_fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Genre",
        yaxis_title="Aantal films",
        yaxis_type="log",  
        width=1200,  
        height=600  
    )

    languages_split = filtered_df['language'].str.split(',').explode().value_counts()
    language_fig = px.bar(languages_split, 
                          x=languages_split.index, 
                          y=languages_split.values, 
                          labels={'x': 'Taal', 'y': 'Aantal films'},
                          title="Aantal films per taal")

    language_fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Taal",
        yaxis_title="Aantal films",
        yaxis_type="log",  
        width=1200,  
        height=600  
    )
    
    avg_budget_per_year = filtered_df.groupby('year')['budget_usd'].mean().reset_index()
    avg_budget_year_fig = px.line(avg_budget_per_year, x='year', y='budget_usd', labels={'year': 'Jaar', 'budget_usd': 'Gemiddeld Budget in USD'}, title="Gemiddeld Budget per Jaar")
    
    table_data = filtered_df[['title', 'year', 'director', 'actors', 'budget_usd', 'avg_vote']].to_dict('records')

    return genre_fig, language_fig, avg_budget_year_fig, table_data

if __name__ == '__main__':
    app.run(debug=True)
