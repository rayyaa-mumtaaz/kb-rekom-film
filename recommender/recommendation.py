"""Satu-satunya tempat yang menyambungkan A* dan Hill Climbing generik dengan data film."""

import random
import sqlite3

from recommender import config, database
from recommender.astar import a_star_search
from recommender.hill_climbing import hill_climbing
from recommender.models import Movie
from recommender.results import (
    ClimbResult,
    Goal,
    PathStep,
    RecommendationResult,
    RecommendedMovie,
)
from recommender.similarity import Features, features, similarity_between


class RecommendationService:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.movies: dict[int, Movie] = {}
        self.features: dict[int, Features] = {}
        self.graph: dict[int, dict[int, float]] = {}  # movie_id -> {neighbor_id: cost}
        self.edge_reasons: dict[tuple[int, int], str] = {}
        self.seed_ids: list[int] = []
        self.heuristic_memo: dict[int, float] = {}  # setiap film dihitung paling banyak sekali per request

    def recommend(self, seed_movies: list[Movie]) -> RecommendationResult:
        self.load_movies_and_graph()
        self.seed_ids = [movie.id for movie in seed_movies]
        self.heuristic_memo = {}

        search = a_star_search(
            starts=self.seed_ids,
            neighbors=lambda node: self.graph.get(node, {}),
            heuristic=self.heuristic,
            is_goal=lambda node: node not in self.seed_ids,
            goal_limit=config.ASTAR["candidate_limit"],
            max_expansions=config.ASTAR["max_expansions"],
        )

        climb = self.select_diverse_set(search.goals)

        return RecommendationResult(
            seeds=seed_movies,
            items=self.build_items(climb.best_state, search.goals),
            search=search,
            climb=climb,
            titles={movie_id: movie.title for movie_id, movie in self.movies.items()},
        )

    def heuristic(self, node: int) -> float:
        """
        h(n) = 1 − rata-rata sim(n, film pilihan).

        Semakin mirip dengan selera pengguna, semakin kecil. Tidak dijamin admissible:
        tujuannya mengurutkan kandidat, bukan mencari jalur terpendek.
        """
        if node not in self.heuristic_memo:
            similarities = [
                similarity_between(self.features[node], self.features[seed_id]) for seed_id in self.seed_ids
            ]
            self.heuristic_memo[node] = 1 - sum(similarities) / len(similarities)

        return self.heuristic_memo[node]

    def select_diverse_set(self, goals: list[Goal]) -> ClimbResult:
        settings = config.HILL_CLIMBING

        # State awal = kandidat dengan f terkecil.
        candidate_ids = [goal.node for goal in sorted(goals, key=lambda goal: goal.f)]
        state_size = min(settings["result_size"], len(candidate_ids))

        relevance = {movie_id: 1 - self.heuristic(movie_id) for movie_id in candidate_ids}
        similarity_matrix = self.similarity_matrix(candidate_ids)

        return hill_climbing(
            pool=candidate_ids,
            initial_state=candidate_ids[:state_size],
            state_size=state_size,
            score=lambda state: self.score(state, relevance, similarity_matrix),
            restarts=settings["restarts"],
            max_iterations=settings["max_iterations"],
            randomizer=random.Random(settings["seed"]),
        )

    @staticmethod
    def score(state: list[int], relevance: dict[int, float], similarity_matrix: dict[tuple[int, int], float]) -> float:
        """
        skor(state) = 0.7 · rata-rata relevansi + 0.3 · keragaman
        keragaman   = rata-rata (1 − sim(a, b)) untuk semua pasangan di state

        Hanya membaca nilai yang sudah dihitung, karena fungsi ini dipanggil ratusan kali per iterasi.
        """
        average_relevance = sum(relevance[movie_id] for movie_id in state) / len(state)

        pairs = [(a, b) for i, a in enumerate(state) for b in state[i + 1 :]]
        diversity = sum(1 - similarity_matrix[pair] for pair in pairs) / len(pairs) if pairs else 0.0

        return (
            config.HILL_CLIMBING["relevance_weight"] * average_relevance
            + config.HILL_CLIMBING["diversity_weight"] * diversity
        )

    def similarity_matrix(self, candidate_ids: list[int]) -> dict[tuple[int, int], float]:
        """sim untuk semua pasangan kandidat dihitung sekali (30 kandidat = 435 pasangan)."""
        matrix = {}

        for i, a in enumerate(candidate_ids):
            for b in candidate_ids[i + 1 :]:
                matrix[(a, b)] = matrix[(b, a)] = similarity_between(self.features[a], self.features[b])

        return matrix

    def build_items(self, selected_ids: list[int], goals: list[Goal]) -> list[RecommendedMovie]:
        path_by_goal = {goal.node: goal.path for goal in goals}

        items = [
            RecommendedMovie(
                movie=self.movies[movie_id],
                relevance=1 - self.heuristic(movie_id),
                path=self.describe_path(path_by_goal[movie_id]),
            )
            for movie_id in selected_ids
        ]

        return sorted(items, key=lambda item: item.relevance, reverse=True)

    def describe_path(self, path: list[int]) -> list[PathStep]:
        return [
            PathStep(self.movies[a], self.movies[b], self.edge_reasons[(a, b)])
            for a, b in zip(path, path[1:], strict=False)
        ]

    def load_movies_and_graph(self) -> None:
        self.movies = database.load_movies(self.connection)
        self.features = {movie_id: features(movie) for movie_id, movie in self.movies.items()}
        self.graph = {}
        self.edge_reasons = {}

        for edge in database.load_edges(self.connection):
            self.graph.setdefault(edge["movie_id"], {})[edge["neighbor_id"]] = edge["cost"]
            self.edge_reasons[(edge["movie_id"], edge["neighbor_id"])] = edge["reason"]
