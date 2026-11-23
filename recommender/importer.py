"""Impor TMDB 5000: ambil film dengan vote_count tertinggi lalu gabungkan dengan credits."""

import csv
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from recommender import config, database
from recommender.models import Movie

# Kolom JSON cast/crew bisa sangat panjang, melebihi batas bawaan modul csv.
csv.field_size_limit(2**31 - 1)


def import_movies(connection: sqlite3.Connection, dataset_dir: Path = config.DATASET_DIR) -> int:
    movies = select_top_movies(dataset_dir / "tmdb_5000_movies.csv", config.MOVIE_LIMIT)
    credits = read_credits(dataset_dir / "tmdb_5000_credits.csv", {movie["id"] for movie in movies})
    rows = disambiguate_titles([merge_credits(movie, credits) for movie in movies])

    with connection:
        connection.execute("DELETE FROM movie_edges")
        connection.execute("DELETE FROM movies")
        for start in range(0, len(rows), 100):
            database.insert_movies(connection, [Movie(**row) for row in rows[start : start + 100]])
            print_progress(min(start + 100, len(rows)), len(rows))

    return connection.execute("SELECT COUNT(*) FROM movies").fetchone()[0]


def select_top_movies(path: Path, limit: int) -> list[dict]:
    movies = [
        {
            "id": int(row["id"]),
            "title": row["title"].strip(),
            "year": int(row["release_date"][:4]) if row["release_date"] else None,
            "vote_average": float(row["vote_average"]),
            "vote_count": int(row["vote_count"]),
            "overview": row["overview"] or None,
            "genres": names(row["genres"]),
            "keywords": names(row["keywords"])[: config.MAX_KEYWORDS],
        }
        for row in read_csv(path)
    ]

    # Urutan id sebagai pemecah seri agar hasil impor selalu sama.
    movies.sort(key=lambda movie: (-movie["vote_count"], movie["id"]))
    return movies[:limit]


def read_credits(path: Path, selected_ids: set[int]) -> dict[int, dict]:
    credits = {}

    for row in read_csv(path):
        movie_id = int(row["movie_id"])
        if movie_id in selected_ids:
            credits[movie_id] = {
                "cast": top_cast(json.loads(row["cast"] or "[]")),
                "director": director(json.loads(row["crew"] or "[]")),
            }

    return credits


def merge_credits(movie: dict, credits: dict[int, dict]) -> dict:
    movie_credits = credits.get(movie["id"], {"cast": [], "director": None})
    return {**movie, **movie_credits}


def disambiguate_titles(movies: list[dict]) -> list[dict]:
    """Judul kembar diberi tahun agar input judul di form tidak ambigu."""
    title_counts = Counter(movie["title"] for movie in movies)

    return [
        {**movie, "title": f"{movie['title']} ({movie['year'] or '?'})"} if title_counts[movie["title"]] > 1 else movie
        for movie in movies
    ]


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    # Dibaca per baris agar file credits (±40 MB) tidak dimuat sekaligus.
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def names(json_text: str) -> list[str]:
    return [item["name"] for item in json.loads(json_text or "[]")]


def top_cast(cast: list[dict]) -> list[str]:
    return [member["name"] for member in sorted(cast, key=lambda member: member["order"])][: config.MAX_CAST]


def director(crew: list[dict]) -> str | None:
    return next((member["name"] for member in crew if member["job"] == "Director"), None)


def print_progress(done: int, total: int) -> None:
    width = 30
    filled = width * done // total
    sys.stdout.write(f"\r{done:>5}/{total} [{'=' * filled}{'-' * (width - filled)}]")
    if done == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
