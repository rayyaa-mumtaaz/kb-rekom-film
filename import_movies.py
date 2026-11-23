"""Jalankan: python import_movies.py"""

import sys

from recommender import config, database
from recommender.importer import import_movies

if __name__ == "__main__":
    for name in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv"):
        if not (config.DATASET_DIR / name).is_file():
            sys.exit(f"File dataset tidak ditemukan: {config.DATASET_DIR / name}")

    connection = database.connect()
    print(f"Jumlah film diimpor: {import_movies(connection)}")
