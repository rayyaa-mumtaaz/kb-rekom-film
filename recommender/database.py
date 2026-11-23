import json
import sqlite3
from pathlib import Path

from recommender import config
from recommender.models import Movie

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,            -- id TMDB, agar movies dan credits bisa digabung langsung
    title TEXT NOT NULL UNIQUE,
    year INTEGER,
    vote_average REAL NOT NULL,
    vote_count INTEGER NOT NULL,
    overview TEXT,
    genres TEXT NOT NULL,              -- JSON array
    "cast" TEXT NOT NULL,              -- JSON array
    director TEXT,
    keywords TEXT NOT NULL             -- JSON array
);

CREATE TABLE IF NOT EXISTS movie_edges (
    movie_id INTEGER NOT NULL REFERENCES movies (id) ON DELETE CASCADE,
    neighbor_id INTEGER NOT NULL REFERENCES movies (id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    cost REAL NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (movie_id, neighbor_id)
);

CREATE INDEX IF NOT EXISTS movie_edges_movie_id ON movie_edges (movie_id);
"""


def connect(path: Path | str = config.DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def insert_movies(connection: sqlite3.Connection, movies: list[Movie]) -> None:
    connection.executemany(
        'INSERT INTO movies (id, title, year, vote_average, vote_count, overview, genres, "cast", director, keywords)'
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                movie.id,
                movie.title,
                movie.year,
                movie.vote_average,
                movie.vote_count,
                movie.overview,
                json.dumps(movie.genres),
                json.dumps(movie.cast),
                movie.director,
                json.dumps(movie.keywords),
            )
            for movie in movies
        ],
    )


def load_movies(connection: sqlite3.Connection) -> dict[int, Movie]:
    rows = connection.execute("SELECT * FROM movies ORDER BY id").fetchall()
    return {row["id"]: _movie_from_row(row) for row in rows}


def load_titles(connection: sqlite3.Connection) -> list[str]:
    return [row["title"] for row in connection.execute("SELECT title FROM movies ORDER BY title")]


def load_edges(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute("SELECT movie_id, neighbor_id, similarity, cost, reason FROM movie_edges").fetchall()


def _movie_from_row(row: sqlite3.Row) -> Movie:
    return Movie(
        id=row["id"],
        title=row["title"],
        year=row["year"],
        vote_average=row["vote_average"],
        vote_count=row["vote_count"],
        overview=row["overview"],
        genres=json.loads(row["genres"]),
        cast=json.loads(row["cast"]),
        director=row["director"],
        keywords=json.loads(row["keywords"]),
    )
