"""
Kemiripan dua film:

    sim(a, b) = 0.40·J(genre) + 0.25·J(aktor) + 0.20·[sutradara sama] + 0.15·J(keyword)

Fitur disimpan sebagai dict (dict.fromkeys) agar bisa dipakai seperti himpunan
untuk menghitung irisan, tetapi urutan anggotanya tetap terjaga untuk teks alasan.
"""

from typing import Any

from recommender import config
from recommender.models import Movie

Features = dict[str, Any]

REASON_LABELS = {
    "genres": "genre sama",
    "cast": "aktor sama",
    "director": "sutradara sama",
    "keywords": "keyword sama",
}


def features(movie: Movie) -> Features:
    return {
        "genres": dict.fromkeys(movie.genres),
        "cast": dict.fromkeys(movie.cast),
        "director": movie.director,
        "keywords": dict.fromkeys(movie.keywords),
    }


def jaccard(set_a: dict | set, set_b: dict | set) -> float:
    """J(A, B) = |A ∩ B| / |A ∪ B|, bernilai 0 jika keduanya kosong."""
    intersection = sum(1 for member in set_a if member in set_b)
    union = len(set_a) + len(set_b) - intersection
    return 0.0 if union == 0 else intersection / union


def contributions(a: Features, b: Features, weights: dict[str, float] = config.WEIGHTS) -> dict[str, float]:
    same_director = a["director"] is not None and a["director"] == b["director"]
    return {
        "genres": weights["genres"] * jaccard(a["genres"], b["genres"]),
        "cast": weights["cast"] * jaccard(a["cast"], b["cast"]),
        "director": weights["director"] * (1.0 if same_director else 0.0),
        "keywords": weights["keywords"] * jaccard(a["keywords"], b["keywords"]),
    }


def similarity_between(a: Features, b: Features, weights: dict[str, float] = config.WEIGHTS) -> float:
    return sum(contributions(a, b, weights).values())


def reason_between(a: Features, b: Features, weights: dict[str, float] = config.WEIGHTS) -> str:
    """Alasan = komponen dengan kontribusi berbobot terbesar, ditulis bersama anggota irisannya."""
    component_scores = contributions(a, b, weights)
    best_component = max(component_scores, key=component_scores.get)

    if component_scores[best_component] <= 0.0:
        return "tidak ada kemiripan"

    if best_component == "director":
        shared_members = [a["director"]]
    else:
        shared_members = [name for name in a[best_component] if name in b[best_component]]

    return f"{REASON_LABELS[best_component]}: {', '.join(shared_members)}"


def similarity(movie_a: Movie, movie_b: Movie, weights: dict[str, float] = config.WEIGHTS) -> float:
    return similarity_between(features(movie_a), features(movie_b), weights)


def reason(movie_a: Movie, movie_b: Movie, weights: dict[str, float] = config.WEIGHTS) -> str:
    return reason_between(features(movie_a), features(movie_b), weights)
