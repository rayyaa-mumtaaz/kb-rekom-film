"""Struktur data hasil (DTO). Hanya menampung data, tanpa logika."""

from dataclasses import dataclass
from typing import Any

from recommender.models import Movie

Node = Any  # int (id film) atau str (graf contoh materi)


@dataclass(frozen=True)
class Goal:
    node: Node
    g: float
    h: float
    f: float
    path: list[Node]  # dari start ke goal


@dataclass(frozen=True)
class Step:
    step: int
    node: Node
    g: float
    h: float
    f: float
    action: str  # "start", "expand", atau "goal"


@dataclass(frozen=True)
class SearchResult:
    goals: list[Goal]  # urut sesuai ditemukan
    steps: list[Step]  # setiap node yang diambil dari Open List
    expanded_count: int
    duration_ms: float


@dataclass(frozen=True)
class ClimbStep:
    iteration: int  # 0 = state awal
    removed: Node | None
    added: Node | None
    score: float


@dataclass(frozen=True)
class ClimbResult:
    best_state: list[Node]
    best_score: float
    best_restart: int
    log: dict[int, list[ClimbStep]]  # per restart, mulai dari 1
    duration_ms: float


@dataclass(frozen=True)
class PathStep:
    from_movie: Movie
    to_movie: Movie
    reason: str


@dataclass(frozen=True)
class RecommendedMovie:
    movie: Movie
    relevance: float
    path: list[PathStep]  # jalur alasan dari film pilihan ke film ini


@dataclass(frozen=True)
class RecommendationResult:
    seeds: list[Movie]
    items: list[RecommendedMovie]  # urut relevansi menurun
    search: SearchResult
    climb: ClimbResult
    titles: dict[int, str]  # id -> judul, untuk menampilkan log yang berisi id
