"""
Steepest-ascent Hill Climbing dengan random restart.

State = himpunan state_size item dari pool. Generik: hanya tahu item dan fungsi skor.
"""

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from recommender.results import ClimbResult, ClimbStep, Node

Score = Callable[[list[Node]], float]


def hill_climbing(
    pool: list[Node],
    initial_state: list[Node],
    state_size: int,
    score: Score,
    restarts: int,
    max_iterations: int,
    randomizer: random.Random,
) -> ClimbResult:
    started_at = time.perf_counter()
    best_state, best_score, best_restart = initial_state, float("-inf"), 1
    log: dict[int, list[ClimbStep]] = {}

    for restart in range(1, restarts + 1):
        # 1. Initial state: restart pertama dari state yang diberikan, berikutnya acak.
        if restart == 1:
            current_state = list(initial_state)
        else:
            current_state = randomizer.sample(pool, state_size)

        current_state, current_score, log[restart] = climb(current_state, pool, score, max_iterations)

        if current_score > best_score:
            best_state, best_score, best_restart = current_state, current_score, restart

    return ClimbResult(
        best_state=best_state,
        best_score=best_score,
        best_restart=best_restart,
        log=log,
        duration_ms=(time.perf_counter() - started_at) * 1000,
    )


def climb(
    current_state: list[Node], pool: list[Node], score: Score, max_iterations: int
) -> tuple[list[Node], float, list[ClimbStep]]:
    """Satu kali pendakian dari current_state sampai local maximum atau batas iterasi."""
    current_score = score(current_state)
    log = [ClimbStep(0, None, None, current_score)]

    for iteration in range(1, max_iterations + 1):
        best_neighbor = find_best_neighbor(current_state, pool, score)

        # 4. Termination: tidak ada neighbor yang lebih baik berarti sudah di local maximum.
        if best_neighbor is None or best_neighbor.score <= current_score:
            break

        # 3. Move to better state.
        current_state, current_score = best_neighbor.state, best_neighbor.score
        log.append(ClimbStep(iteration, best_neighbor.removed, best_neighbor.added, current_score))

    return current_state, current_score, log


@dataclass(frozen=True, slots=True)
class Neighbor:
    state: list[Node]
    removed: Node
    added: Node
    score: float


def find_best_neighbor(current_state: list[Node], pool: list[Node], score: Score) -> Neighbor | None:
    """Steepest ascent: semua neighbor dinilai, yang skornya tertinggi dipilih."""
    best_neighbor = None

    for state, removed, added in neighbors(current_state, pool):
        neighbor_score = score(state)

        if best_neighbor is None or neighbor_score > best_neighbor.score:
            best_neighbor = Neighbor(state, removed, added, neighbor_score)

    return best_neighbor


def neighbors(current_state: list[Node], pool: list[Node]) -> Iterator[tuple[list[Node], Node, Node]]:
    """2. Neighboring states: tukar 1 item di dalam state dengan 1 item di luar state."""
    outside = [item for item in pool if item not in current_state]

    for index, removed in enumerate(current_state):
        for added in outside:
            state = list(current_state)
            state[index] = added
            yield state, removed, added
