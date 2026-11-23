"""Semua angka parameter sistem rekomendasi ada di sini, bukan di kode algoritma."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATABASE_PATH = ROOT / "data" / "recommender.sqlite"

MOVIE_LIMIT = 1000  # jumlah film diimpor (vote_count tertinggi)
MAX_CAST = 3
MAX_KEYWORDS = 10
NEIGHBORS_PER_MOVIE = 10  # tetangga per node di graf

# Bobot sim(a, b), total 1.0.
WEIGHTS = {
    "genres": 0.40,
    "cast": 0.25,
    "director": 0.20,
    "keywords": 0.15,
}

ASTAR = {
    "candidate_limit": 30,  # jumlah goal yang dikumpulkan
    "max_expansions": 500,  # batas aman ekspansi node
}

HILL_CLIMBING = {
    "result_size": 10,
    "relevance_weight": 0.7,
    "diversity_weight": 0.3,
    "restarts": 5,
    "max_iterations": 100,
    "seed": 42,
}

SEEDS = {"min": 3, "max": 5}  # jumlah film input pengguna
