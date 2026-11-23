"""Graf kemiripan: setiap film terhubung ke film lain yang paling mirip, cost = 1 − sim."""

import sqlite3

from recommender import config, database
from recommender.importer import print_progress
from recommender.similarity import Features, features, reason_between, similarity_between


def build_graph(connection: sqlite3.Connection) -> int:
    # Fitur disiapkan sekali agar loop 1.000 × 1.000 hanya bekerja di atas dict biasa.
    all_features = {movie_id: features(movie) for movie_id, movie in database.load_movies(connection).items()}

    with connection:
        connection.execute("DELETE FROM movie_edges")

        for done, (movie_id, movie_features) in enumerate(all_features.items(), start=1):
            edges = edges_for(movie_id, movie_features, all_features, config.NEIGHBORS_PER_MOVIE)
            connection.executemany(
                "INSERT INTO movie_edges (movie_id, neighbor_id, similarity, cost, reason) VALUES (?, ?, ?, ?, ?)",
                edges,
            )
            print_progress(done, len(all_features))

    return connection.execute("SELECT COUNT(*) FROM movie_edges").fetchone()[0]


def edges_for(
    movie_id: int, movie_features: Features, all_features: dict[int, Features], neighbors_per_movie: int
) -> list[tuple[int, int, float, float, str]]:
    scores = {
        other_id: similarity_between(movie_features, other_features)
        for other_id, other_features in all_features.items()
        if other_id != movie_id
    }

    # sorted() stabil, jadi film dengan sim sama tetap urut id: hasil build selalu sama.
    top_neighbors = sorted(
        ((other_id, sim) for other_id, sim in scores.items() if sim > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:neighbors_per_movie]

    return [
        (movie_id, neighbor_id, sim, 1 - sim, reason_between(movie_features, all_features[neighbor_id]))
        for neighbor_id, sim in top_neighbors
    ]
