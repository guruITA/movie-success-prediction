"""
Dashboard IMDb Dataset using Dash
===================================

This Dash application provides an interactive dashboard for an IMDb dataset. The application includes:
  - An Overview page showing all movies with multiple graphical representations (budget, rating, language, and average budget per year).
  - A Details page that follows the Visual Information Seeking Mantra to drill down into a specific movie.
    In this page, director and actor names are clickable and lead to a page with all movies related to that person.
  - A Prediction page that accepts new movie information and uses a trained machine learning model to forecast movie success.
    In this page, you can select from existing directors, genres, countries, and languages.

The graphs are styled using Plotly’s "simple_white" template in adherence with Tufte’s guidelines for minimal, clear visualizations.
The data source used is the IMDb dataset stored in "data/cleaned_movies.parquet" (simulating a SQL/NoSQL database).

Literature References:
[1] Shneiderman, B. (1996). “The eyes have it: a task by data type taxonomy for information visualizations.”
[2] Guidelines for Good Visual Information Representations, Interaction Design Foundation.
[3] Choosing Colors, Masnick.

Author: [Your Name]
Date: [Today's Date]
"""

import os
import json
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from urllib.parse import parse_qs
import re
import logging
from flask_caching import Cache
from sklearn.linear_model import LinearRegression

# Enable logging
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Create the Dash app instance
# ---------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# ---------------------------
# Global styling settings
# ---------------------------
global_style = {
    'backgroundColor': '#141414',
    'color': '#ffffff',
    'font-family': 'Arial, sans-serif',
    'padding': '20px'
}

# Grafieken krijgen nu een iets bredere breedte
graph_style = {
    'width': 900,
    'height': 400
}

# ---------------------------
# Initialize permanent caching using FileSystemCache.
# ---------------------------
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache',
    'CACHE_DEFAULT_TIMEOUT': 0  
})
cache.init_app(app.server)

# ---------------------------
# Utility: Financial Currency Extraction
# ---------------------------
CURRENCY_CONVERSION = {
    "AED": 0.27229408, "ALL": 0.010607085, "AMD": 0.0025354709, "ARS": 0.00094495163, "ATS": 0.08,
    "AUD": 0.63535408, "AZM": 0.00011764706, "BDT": 0.0082125132, "BEF": 0.025937687, "BGL": 0.53,
    "BND": 0.74841975, "BRL": 0.17441155, "CAD": 0.70258484, "CHF": 1.1138894, "CLP": 0.0010610028,
    "CNY": 0.13792966, "COP": 0.00024432851, "CZK": 0.041720216, "DEM": 0.53497681, "DKK": 0.14025151,
    "DOP": 0.016090042, "EEK": 0.066872101, "EGP": 0.019769664, "ESP": 0.0062885321, "EUR": 1.0463237,
    "FIM": 0.17597901, "FRF": 0.15951102, "GBP": 1.2634503, "GEL": 0.35651896, "GRD": 0.0030706491,
    "HKD": 0.12870172, "HRK": 0.13887102, "HUF": 0.0025946953, "IDR": 0.000061242798, "IEP": 1.3285924,
    "ILS": 0.28038205, "INR": 0.011547487, "IRR": 0.000023808041, "ISK": 0.0071811051, "ITL": 0.00054039547,
    "JMD": 0.0063598821, "JPY": 0.0066990678, "KRW": 0.00069711919, "LKR": 0.0033819768, "LTL": 0.30304435,
    "LVL": 1.4888326, "MNT": 0.0002888036, "MTL": 2.4373435, "MXN": 0.048939409, "MYR": 0.22630846,
    "NGN": 0.00066686304, "NLG": 0.47481363, "NOK": 0.089717847, "NPR": 0.0072137981, "NZD": 0.57405229,
    "PHP": 0.017278612, "PKR": 0.0035780806, "PLN": 0.25140371, "PTE": 0.0052191795, "PYG": 0.00012609691,
    "ROL": 0.000021016896, "RON": 0.21016896, "RUR": 0.000011299218, "SEK": 0.09389212, "SGD": 1.0,
    "SIT": 0.004366515, "SKK": 0.004366515, "THB": 0.004366515, "TRL": 0.000000027448082, "TTD": 0.14717958,
    "TWD": 0.030472035, "UAH": 0.023940279, "VEB": 0.00000000015790072, "VND": 0.000039192881, "XAU": 2936.0832,
    "YUM": 0.54, "ZAR": 0.054467969, "$": 1.00
}

def clean_currency(value):
    if not isinstance(value, str) or pd.isna(value) or value.strip() == "":
        return None
    parts = value.split(maxsplit=1)
    if len(parts) == 2:
        currency, amount = parts
    elif len(parts) == 1 and parts[0][0] in "$ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        currency, amount = parts[0][0], parts[0][1:]
    else:
        logging.warning(f"Skipping unknown format: {value}")
        return None
    try:
        amount = float(amount.replace(",", ""))
    except ValueError:
        return None
    conversion_rate = CURRENCY_CONVERSION.get(currency.strip(), None)
    if conversion_rate is not None:
        return round(amount * conversion_rate, 2)
    else:
        logging.warning(f"Unknown currency found: {currency}")
        return None

def clean_financial_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df[columns] = df[columns].map(clean_currency)
    return df

# ---------------------------
# Data Cleaning and Success Calculation
# ---------------------------
def clean_movie_data_for_success(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["budget"].dtype == object:
        df["budget"] = df["budget"].apply(clean_currency)
    else:
        df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    if df["worlwide_gross_income"].dtype == object:
        df["worlwide_gross_income"] = df["worlwide_gross_income"].apply(clean_currency)
    if df["usa_gross_income"].dtype == object:
        df["usa_gross_income"] = df["usa_gross_income"].apply(clean_currency)
    df = df.dropna(subset=["budget"])
    df["corrected_income"] = df["worlwide_gross_income"].fillna(df["usa_gross_income"])
    df["corrected_income"] = df["corrected_income"].fillna(df["budget"])
    df["avg_vote"] = pd.to_numeric(df["avg_vote"], errors="coerce")
    df = df.dropna(subset=["avg_vote"])
    df = df[df["budget"] >= 1000]
    return df

def compute_composite_success(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["budget", "corrected_income", "avg_vote"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["financial_success"] = (df["corrected_income"] - df["budget"]) / df["budget"]
    df["vote_success"] = df["avg_vote"] / 10
    df["composite_success"] = 0.7 * df["financial_success"] + 0.3 * df["vote_success"]
    df["composite_success_scaled"] = np.tanh(df["composite_success"].astype(np.float64))
    return df

# ---------------------------
# Data Loading & Preprocessing for Dash
# ---------------------------
df = pd.read_parquet("data/cleaned_movies.parquet")

# 1. Talen opsplitsen
df_language = df.copy()
df_language['language'] = df_language['language'].fillna("").str.split(',')
df_language = df_language.explode('language')
df_language['language'] = df_language['language'].str.strip()
language_counts = df_language['language'].value_counts().reset_index()
language_counts.columns = ['language', 'count']
budget_by_year = df.groupby("year")["budget"].mean().reset_index()

# 2. Genres opsplitsen
df_genre = df.copy()
df_genre['genre'] = df_genre['genre'].fillna("").str.split(',')
df_genre = df_genre.explode('genre')
df_genre['genre'] = df_genre['genre'].str.strip()
unique_genres = sorted(df_genre['genre'].unique())

# 3. Acteurs opsplitsen
df_actors = df.copy()
df_actors['actors'] = df_actors['actors'].fillna("").str.split(',')
df_actors = df_actors.explode('actors')
df_actors['actors'] = df_actors['actors'].str.strip()
unique_actors = sorted(df_actors['actors'].unique())

# 4. Directors opsplitsen
df_directors = df.copy()
df_directors['director'] = df_directors['director'].fillna("").str.split(',')
df_directors = df_directors.explode('director')
df_directors['director'] = df_directors['director'].str.strip()
unique_directors = sorted(df_directors['director'].dropna().unique())

# 5. Landen opsplitsen
df_country = df.copy()
df_country["country"] = df_country["country"].fillna("").str.split(',')
df_country = df_country.explode("country")
df_country["country"] = df_country["country"].str.strip()
unique_countries = sorted(df_country["country"].dropna().unique())

# 6. Gebruik de opgesplitste talen
unique_languages = sorted(df_language['language'].dropna().unique())

# ---------------------------
# JSON Buffer Functions for Dropdown Options (Preload at startup)
# ---------------------------
def get_dropdown_options(filename, compute_fn):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            options = json.load(f)
            logging.info(f"Loaded {filename} from disk.")
    else:
        options = compute_fn()
        with open(filename, 'w') as f:
            json.dump(options, f)
            logging.info(f"Computed and saved {filename}.")
    return options

@cache.memoize()
def compute_directors():
    valid_directors = [d for d in unique_directors if isinstance(d, str) and d.strip() != ""]
    return [{"label": d, "value": d} for d in valid_directors]

@cache.memoize()
def compute_actors():
    top_actors = df_actors['actors'].value_counts().head(50).index.tolist()
    valid_actors = [a for a in top_actors if isinstance(a, str) and a.strip() != ""]
    return [{"label": a, "value": a} for a in sorted(valid_actors)]

@cache.memoize()
def compute_genres():
    valid_genres = [g for g in unique_genres if isinstance(g, str) and g.strip() != ""]
    return [{"label": g, "value": g} for g in valid_genres]

@cache.memoize()
def compute_countries():
    valid_countries = [c for c in unique_countries if isinstance(c, str) and c.strip() != ""]
    return [{"label": c, "value": c} for c in valid_countries]

@cache.memoize()
def compute_languages():
    valid_languages = [l for l in unique_languages if isinstance(l, str) and l.strip() != ""]
    return [{"label": l, "value": l} for l in valid_languages]

def preload_options():
    dropdown_directors = get_dropdown_options("directors.json", compute_directors)
    dropdown_actors = get_dropdown_options("actors.json", compute_actors)
    dropdown_genres = get_dropdown_options("genres.json", compute_genres)
    dropdown_countries = get_dropdown_options("countries.json", compute_countries)
    dropdown_languages = get_dropdown_options("languages.json", compute_languages)
    return dropdown_directors, dropdown_actors, dropdown_genres, dropdown_countries, dropdown_languages

dropdown_directors, dropdown_actors, dropdown_genres, dropdown_countries, dropdown_languages = preload_options()

# ---------------------------
# Dummy Target Encoding Mapping & Real Prediction Pipeline
# ---------------------------
target_mapping = {
    "director": {d: 0.5 for d in unique_directors},
    "actors": {a: 0.5 for a in unique_actors},
    "genre": {g: 0.5 for g in unique_genres},
    "country": {c: 0.5 for c in unique_countries},
    "language": {l: 0.5 for l in unique_languages}
}
numeric_feature = "budget"
categorical_features = ["director", "actors", "genre", "country", "language"]

def encode_new_movie_features(new_movie, target_mapping):
    numeric_val = np.log(float(new_movie.get(numeric_feature, 0)) + 1)
    cat_values = []
    for feature in categorical_features:
        val = new_movie.get(feature, "")
        if val:
            tokens = [t.strip() for t in val.split(",")]
            default_val = 0.5 if feature == "actors" else 0.0
            mapping = target_mapping.get(feature, {})
            token_vals = [mapping.get(token, default_val) for token in tokens]
            cat_values.append(np.mean(token_vals) if token_vals else default_val)
        else:
            cat_values.append(0.0)
    return np.array([numeric_val] + cat_values)

def predict_movie_success(new_movie, model, target_mapping):
    features = encode_new_movie_features(new_movie, target_mapping)
    features = features.reshape(1, -1)
    return model.predict(features)[0]

def predict_new_movie_success(new_movie, models, target_mapping):
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = predict_movie_success(new_movie, model, target_mapping)
    return predictions

# ---------------------------
# Real Model Training (Simplified)
# ---------------------------
df_clean = df.copy()
df_clean = clean_movie_data_for_success(df_clean)
df_clean = compute_composite_success(df_clean)
features_list = df_clean.apply(lambda row: encode_new_movie_features(row, target_mapping), axis=1)
X = np.vstack(features_list.values)
y = df_clean["composite_success_scaled"].values
lr_model = LinearRegression()
lr_model.fit(X, y)
trained_models = {"Linear Regression": lr_model}
logging.info("Trained Linear Regression model for prediction.")

# ---------------------------
# Dash Application Global Layout
# ---------------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style=global_style)

# ---------------------------
# Page Layouts
# ---------------------------
def create_overview_layout():
    fig_budget = px.histogram(df, x="budget", nbins=30, title="Budget Distribution")
    fig_budget.update_layout(template="simple_white", width=graph_style['width'], height=graph_style['height'])
    
    fig_rating = px.histogram(df, x="avg_vote", nbins=30, title="Rating Distribution")
    fig_rating.update_layout(template="simple_white", width=graph_style['width'], height=graph_style['height'])
    
    genres = df['genre'].dropna().str.split(',').explode().str.strip().str.lower()
    genre_count = genres.value_counts().reset_index()
    genre_count.columns = ['Genre', 'Aantal films']
    genre_fig = px.bar(genre_count, x='Genre', y='Aantal films', title="Aantal films per genres")
    genre_fig.update_layout(xaxis_tickangle=-45, yaxis_type="log",
                            width=graph_style['width'], height=graph_style['height'])
    
    languages_split = df['language'].dropna().str.split(',').explode().str.strip()
    language_count = languages_split.value_counts().reset_index()
    language_count.columns = ['Taal', 'Aantal films']
    language_fig = px.bar(language_count, x='Taal', y='Aantal films', title="Aantal films per taal")
    language_fig.update_layout(xaxis_tickangle=-45, yaxis_type="log",
                               width=graph_style['width'], height=graph_style['height'])
    
    avg_budget_per_year = df.groupby('year')['budget'].mean().reset_index()
    avg_budget_year_fig = px.line(avg_budget_per_year, x='year', y='budget', title="Gemiddeld Budget per Jaar")
    avg_budget_year_fig.update_layout(template="simple_white", width=graph_style['width'], height=graph_style['height'])
    
    layout = html.Div([
        html.H1("Dashboard", style={'textAlign': 'center'}),
        html.A("Prediction", href="/prediction", style={'color': '#f1c40f'}),
        
        html.H2("Films"),
        dash_table.DataTable(
            id='movies-table',
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_data={'color': 'black'},        
            style_header={'color': 'black'}         
        ),
        html.Br(),

        html.Div([
            html.Div([
                html.H3("Filters", style={'margin-bottom': '20px'}),
                dcc.Dropdown(
                    id='title-filter',
                    options=[{'label': t, 'value': t} for t in df['title'].dropna().unique()],
                    multi=True,
                    placeholder="Selecteer filmtitel",
                    style={'margin-bottom': '20px', 'color': 'black'}
                ),
                dcc.Dropdown(
                    id='genre-filter',
                    options=[{'label': g, 'value': g} for g in df['genre'].dropna().str.split(',').explode().unique()],
                    multi=True,
                    placeholder="Selecteer genres",
                    style={'margin-bottom': '20px', 'color': 'black'}
                ),
                html.Div([
                    html.Label("Selecteer Jaarperiode", style={'font-size': '16px'}),
                    dcc.DatePickerRange(
                        id='year-picker',
                        start_date=str(df['year'].min()),
                        end_date=str(df['year'].max()),
                        display_format='YYYY',
                        start_date_placeholder_text="Startjaar",
                        end_date_placeholder_text="Eindjaar",
                        style={'margin-bottom': '20px'}
                    ),
                ]),
                dcc.Dropdown(
                    id='language-filter',
                    options=dropdown_languages,
                    multi=True,
                    placeholder="Selecteer taal",
                    style={'margin-bottom': '20px', 'color': 'black'}
                ),
                dcc.Dropdown(
                    id='actor-filter',
                    options=[{'label': a, 'value': a} for a in df['actors'].dropna().str.split(',').explode().unique()],
                    multi=True,
                    placeholder="Selecteer acteur",
                    style={'margin-bottom': '20px', 'color': 'black'}
                ),
                dcc.Dropdown(
                    id='director-filter',
                    options=dropdown_directors,
                    multi=True,
                    placeholder="Selecteer directeur",
                    style={'margin-bottom': '20px', 'color': 'black'}
                ),
            ], style={'width': '30%', 'padding': '20px'}),
            html.Div([
                dcc.Graph(id='genre-chart', figure=genre_fig),
                dcc.Graph(id='language-chart', figure=language_fig),
                dcc.Graph(id='avg-budget-year-chart', figure=avg_budget_year_fig),
            ], style={'width': '70%'})
        ], style={'display': 'flex'})
    ])
    return layout

def create_detail_layout(movie_id):
    movie = df[df["imdb_title_id"] == movie_id]
    if movie.empty:
        return html.Div([html.H2("Movie not found.")])
    movie = movie.iloc[0]
    director_link = html.A(movie["director"], href=f"/movies_by_person?name={movie['director']}", style={'color': '#f1c40f'})
    actors_list = movie["actors"].split(",")
    actors_links = [html.A(a.strip(), href=f"/movies_by_person?name={a.strip()}", style={'color': '#f1c40f'}) for a in actors_list]
    layout = html.Div([
        html.H1(f"Details for {movie['title']}", style={'textAlign': 'center'}),
        html.P(f"Original Title: {movie['original_title']}"),
        html.P(f"Year: {movie['year']}"),
        html.P(f"Date Published: {movie['date_published']}"),
        html.P(f"Genre: {movie['genre']}"),
        html.P(f"Duration: {movie['duration']} minutes"),
        html.P(f"Country: {movie['country']}"),
        html.P(["Director: ", director_link]),
        html.P(["Actors: ", html.Span(children=actors_links)]),
        html.P(f"Description: {movie['description']}"),
        html.P(f"Average Vote: {movie['avg_vote']}"),
        html.P(f"Votes: {movie['votes']}"),
        html.P(f"Budget: {movie['budget']}"),
        html.P(f"USA Gross Income: {movie['usa_gross_income']}"),
        html.P(f"Worldwide Gross Income: {movie['worlwide_gross_income']}"),
        html.P(f"Metascore: {movie['metascore']}"),
        html.P(f"Reviews from Users: {movie['reviews_from_users']}"),
        html.P(f"Reviews from Critics: {movie['reviews_from_critics']}"),
        html.Br(),
        html.A("Back to Overview", href="/", style={'color': '#f1c40f'})
    ])
    return layout

def create_movies_by_person_layout(person_name):
    filtered_df = df[
        (df["director"].fillna("").str.contains(person_name, case=False)) |
        (df["actors"].fillna("").str.contains(person_name, case=False))
    ]
    layout = html.Div([
        html.H1(f"Movies related to {person_name}", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='person-movies-table',
            columns=[{"name": col, "id": col} for col in filtered_df.columns],
            data=filtered_df.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_data={'color': 'black'},
            style_header={'color': 'black'}
        ),
        html.Br(),
        html.A("Back to Overview", href="/", style={'color': '#f1c40f'})
    ])
    return layout

def create_prediction_layout():
    layout = html.Div([
        html.H1("Prediction", style={'textAlign': 'center'}),
        html.P("Enter details for a new movie to predict its composite success score. "
               "The composite score is computed as 30% from log(budget + 1) and 70% from the average target encoding of director, genre, country, and language, "
               "then normalized using tanh to yield a score between -1 and 1. A positive score indicates predicted success.",
               style={'textAlign': 'center'}),
        html.Div([
            html.Label("Title"),
            dcc.Input(id="new-title", type="text", value="", style={'width': '100%'})
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Director"),
            dcc.Dropdown(
                id="new-director",
                options=dropdown_directors,
                placeholder="Select a director",
                clearable=True,
                style={'color': 'black'}
            )
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Genre (select one or more)"),
            dcc.Dropdown(
                id="new-genre",
                options=dropdown_genres,
                placeholder="Select genre(s)",
                multi=True,
                style={'color': 'black'}
            )
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Country"),
            dcc.Dropdown(
                id="new-country",
                options=dropdown_countries,
                placeholder="Select a country",
                clearable=True,
                style={'color': 'black'}
            )
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Language"),
            dcc.Dropdown(
                id="new-language",
                options=dropdown_languages,
                placeholder="Select a language",
                clearable=True,
                style={'color': 'black'}
            )
        ], style={'margin-bottom': '20px'}),
        html.Div([
            html.Label("Budget (in USD)"),
            dcc.Input(id="new-budget", type="number", value=0, style={'width': '100%'})
        ], style={'margin-bottom': '20px'}),
        html.Button("Predict Success", id="predict-button", style={
            'backgroundColor': '#f1c40f', 
            'color': '#141414', 
            'border': 'none', 
            'padding': '10px 20px', 
            'fontSize': '16px'
        }),
        html.Br(), html.Br(),
        html.Div(id="prediction-output"),
        html.Br(),
        html.A("Back to Overview", href="/", style={'color': '#f1c40f'})
    ], style={'maxWidth': '600px', 'margin': 'auto'})
    return layout

# ---------------------------
# Callbacks
# ---------------------------
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    if not pathname:
        pathname = '/'
    if not search:
        search = ''
    query = parse_qs(search[1:]) if search.startswith('?') else {}
    if pathname == '/':
        return create_overview_layout()
    elif pathname.startswith("/movie/"):
        movie_id = pathname.split("/movie/")[1]
        return create_detail_layout(movie_id)
    elif pathname == "/movies_by_person":
        person_name = query.get("name", [""])[0]
        return create_movies_by_person_layout(person_name)
    elif pathname == "/prediction":
        return create_prediction_layout()
    else:
        return html.Div([
            html.H1("404 - Page not found"),
            html.A("Back to Overview", href="/", style={'color': '#f1c40f'})
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
     Input('language-filter', 'value'),
     Input('actor-filter', 'value'),
     Input('director-filter', 'value')]
)
def update_overview_graphs(selected_titles, selected_genres, start_date, end_date,
                           selected_languages, selected_actors, selected_directors):
    filtered_df = df.copy()
    filtered_df['director'] = filtered_df['director'].astype(str).fillna('')
    filtered_df['actors'] = filtered_df['actors'].astype(str).fillna('')
    filtered_df['language'] = filtered_df['language'].astype(str).fillna('')
    
    if selected_titles:
        filtered_df = filtered_df[filtered_df['title'].isin(selected_titles)]
    
    if selected_genres:
        selected_genres_set = set(g.lower().strip() for g in selected_genres)
        filtered_df = filtered_df[filtered_df['genre'].dropna().str.split(',').apply(
            lambda gs: any(g.strip().lower() in selected_genres_set for g in gs)
        )]
    
    if selected_languages:
        selected_languages_set = set(l.lower().strip() for l in selected_languages)
        filtered_df = filtered_df[filtered_df['language'].apply(
            lambda x: any(lang.strip().lower() in selected_languages_set for lang in x.split(','))
        )]
    
    if selected_actors:
        selected_actors_set = set(a.lower().strip() for a in selected_actors)
        filtered_df = filtered_df[filtered_df['actors'].apply(
            lambda x: any(a.strip().lower() in selected_actors_set for a in x.split(','))
        )]
    
    if selected_directors:
        selected_directors_set = set(d.lower().strip() for d in selected_directors)
        filtered_df = filtered_df[filtered_df['director'].apply(
            lambda x: any(d.strip().lower() in selected_directors_set for d in x.split(','))
        )]
    
    if start_date and end_date:
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        filtered_df = filtered_df[(filtered_df['year'] >= start_year) & (filtered_df['year'] <= end_year)]
    
    genres = filtered_df['genre'].dropna().str.split(',').explode().str.strip().str.lower()
    genre_count = genres.value_counts().reset_index()
    genre_count.columns = ['Genre', 'Aantal films']
    genre_fig = px.bar(genre_count, x='Genre', y='Aantal films', title="Aantal films per genres")
    genre_fig.update_layout(xaxis_tickangle=-45, yaxis_type="log",
                            width=graph_style['width'], height=graph_style['height'])
    
    languages = filtered_df['language'].dropna().str.split(',').explode().str.strip()
    language_count = languages.value_counts().reset_index()
    language_count.columns = ['Taal', 'Aantal films']
    language_fig = px.bar(language_count, x='Taal', y='Aantal films', title="Aantal films per taal")
    language_fig.update_layout(xaxis_tickangle=-45, yaxis_type="log",
                               width=graph_style['width'], height=graph_style['height'])
    
    if not filtered_df.empty:
        avg_budget_per_year = filtered_df.groupby('year')['budget'].mean().reset_index()
        avg_budget_year_fig = px.line(avg_budget_per_year, x='year', y='budget', title="Gemiddeld Budget per Jaar")
        avg_budget_year_fig.update_layout(template="simple_white",
                                          width=graph_style['width'], height=graph_style['height'])
    else:
        avg_budget_year_fig = px.line(title="Gemiddeld Budget per Jaar")
        avg_budget_year_fig.update_layout(width=graph_style['width'], height=graph_style['height'])
    
    table_data = filtered_df.to_dict('records')
    
    return genre_fig, language_fig, avg_budget_year_fig, table_data

@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("new-title", "value"),
     State("new-director", "value"),
     State("new-genre", "value"),
     State("new-country", "value"),
     State("new-language", "value"),
     State("new-budget", "value")]
)
def update_prediction(n_clicks, title, director, genre, country, language, budget):
    if n_clicks is None:
        return ""
    new_movie = {
        "imdb_title_id": "tt_new",
        "title": title or "",
        "director": director if director else "",
        "genre": ", ".join(genre) if genre else "",
        "country": country if country else "",
        "language": language if language else "",
        "budget": budget if budget else 0
    }
    predictions = predict_new_movie_success(new_movie, trained_models, target_mapping)
    explanation = (
        "Explanation:\n"
        "1. The raw composite score is computed as:\n"
        "   0.3 * log(budget + 1) + 0.7 * (average target encoding of director, genre, country, language).\n"
        "2. This raw score is normalized using the tanh function, scaling it to the range [-1, 1].\n"
        "3. A positive tanh-scaled score indicates movie success.\n"
        "   The predictions from the model(s) are shown below."
    )
    output_components = [html.H3("Prediction Results:")]
    for model_name, score in predictions.items():
        output_components.append(html.P(f"{model_name}: Composite Score (tanh-scaled): {score:.4f}"))
    output_components.append(html.Pre(explanation))
    return html.Div(output_components)

# ---------------------------
# Run the App
# ---------------------------
if __name__ == '__main__':
    dropdown_directors, dropdown_actors, dropdown_genres, dropdown_countries, dropdown_languages = preload_options()
    logging.info("Dropdown options preloaded (from JSON files on disk).")
    
    df_clean = df.copy()
    df_clean = clean_movie_data_for_success(df_clean)
    df_clean = compute_composite_success(df_clean)
    features_list = df_clean.apply(lambda row: encode_new_movie_features(row, target_mapping), axis=1)
    X = np.vstack(features_list.values)
    y = df_clean["composite_success_scaled"].values
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    trained_models = {"Linear Regression": lr_model}
    logging.info("Trained Linear Regression model for prediction.")
    
    print("Dash app is running on http://127.0.0.1:8050")
    app.run(debug=True)
