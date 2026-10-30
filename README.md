# Movie Success Prediction: IMDb dashboard with a success score

An analysis of about 81,000 IMDb movies with an interactive Dash dashboard. The dashboard also predicts a success score for a new movie with a machine learning model. Gurpreet Singh made it as the onboarding assignment and workshop exercises of an HvA (Amsterdam University of Applied Sciences) course in spring 2025.

- Team: Gurpreet Singh
- Period: January to April 2025

## What it does

- **Overview page:** a table of the 12,358 movies that have a budget, filters (title, genre, years, language, actor, director) and charts of movies per genre, movies per language and the average budget per year.
- **Detail pages:** `/movie/<imdb id>` shows one movie, for example `/movie/tt0068646`. Click the director or an actor to see all their movies.
- **Prediction page:** pick a director, genres, a country, a language and a budget. A linear regression model gives a success score between -1 and 1. The score is 70% profit compared to the budget and 30% rating, scaled with `tanh`.
- **Scripts:** clean the raw IMDb data, train three models (linear regression, random forest, SGD) and save them as `.pkl` files. Some scripts store the data in a MySQL database. You can skip these: the dashboard works without them.
- **Workshop exercises:** small pandas and matplotlib analyses of football results, crimes and Netflix titles.

## What is where

```text
data/                           CSV and Parquet data (some files use Git LFS)
src/onboarding assigment/       the project code (the folder name has a typo and a space)
  main_v8.py                    the dashboard
  app.py                        an older, simpler dashboard
  movies.py                     cleans data/IMDb movies.csv
  machine_learning_success.py   trains the success models (needs MySQL)
  machine_learning_rating.py    trains models on the rating (reads .env)
  database/connection.py        puts the movies in MySQL and runs example queries
src/workshop 2 exercise/        football exercise
src/workshop 3 exercise/        crimes and Netflix exercises
*.json                          saved dropdown lists for the dashboard
*.pkl, target_mapping.json      models and encoding saved by machine_learning_success.py
requirements.txt                Python packages
.env.example                    database settings for the MySQL scripts
```

## What you need

- Python 3.13 (tested with 3.13.16). The pinned packages do not install on Python 3.14.
- Git with Git LFS (tested with Git 2.53 and git-lfs 3.8). Git for Windows includes Git LFS.
- Only for the MySQL scripts: Docker (tested with Docker Engine 29) or your own MySQL 8 server.

## Getting started

Run every command from the repository root. The scripts read `data/...` relative to the folder you are in.

### 1. Clone the repository

Turn on Git LFS before you clone, or the large data files arrive as small pointer files.

```bash
git lfs install
git clone https://github.com/guruITA/movie-success-prediction.git
cd movie-success-prediction
```

If you cloned without Git LFS, run `git lfs pull` in the repository.

### 2. Make a virtual environment and install the packages

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Start the dashboard

```bash
python "src/onboarding assigment/main_v8.py"
```

The first start takes about half a minute. The app trains its model at startup, and the debug mode loads the file twice. Wait for `Dash is running on http://127.0.0.1:8050/`.

### 4. Open it in the browser

Go to <http://127.0.0.1:8050>. Click **Prediction** at the top for the prediction page, or go to <http://127.0.0.1:8050/prediction>. Stop the app with `Ctrl+C`.

## Other scripts

You can run these after step 2, also from the repository root. The scripts that make charts open a window and wait until you close it.

```bash
python "src/onboarding assigment/app.py"
python "src/onboarding assigment/movies.py"
python "src/workshop 2 exercise/football.py"
python "src/workshop 3 exercise/crimes.py"
python "src/workshop 3 exercise/netflix.py"
```

- `app.py` is the older dashboard, also on <http://127.0.0.1:8050>. Stop `main_v8.py` first.
- `movies.py` writes `data/clean_movies_successfull.csv` again.

### The MySQL scripts (optional)

`database/connection.py`, `machine_learning_success.py` and `machine_learning_rating.py` read database settings from a `.env` file. During the course they used a school database. Use a local MySQL in Docker instead:

```bash
docker run -d --name movie-mysql -e MYSQL_ROOT_PASSWORD=change-me -e MYSQL_DATABASE=movies -p 3306:3306 mysql:8.4
cp .env.example .env
```

`.env.example` already matches this container. Wait about 20 seconds until MySQL is ready, then run:

```bash
python "src/onboarding assigment/database/connection.py"
python "src/onboarding assigment/machine_learning_success.py"
python "src/onboarding assigment/machine_learning_rating.py"
```

- `connection.py` puts the cleaned movies in table `movies` and prints the results of some queries.
- `machine_learning_success.py` writes the features to table `movie_prediction_features`, trains the three models and shows their charts. After you close the chart window it overwrites `lr_model.pkl`, `rf_model.pkl`, `sgd_model.pkl` and `target_mapping.json`. Each of these scripts runs for about a minute.
- `machine_learning_rating.py` trains the same three models on the rating and shows their charts.

Remove the container and its data when you are done:

```bash
docker rm -f -v movie-mysql
```

## Who did what

- **Gurpreet Singh** built everything: the data cleaning (`movies.py`), the MySQL import and queries (`database/connection.py`), the success and rating models (`machine_learning_success.py`, `machine_learning_rating.py`), both Dash dashboards (`app.py`, `main_v8.py`) and the workshop exercises.
