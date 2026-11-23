"""Jalankan: python build_graph.py (setelah import_movies.py)"""

from recommender import database
from recommender.graph import build_graph

if __name__ == "__main__":
    connection = database.connect()
    print(f"Jumlah edge: {build_graph(connection)}")
