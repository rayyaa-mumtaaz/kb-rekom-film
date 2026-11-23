"""
A* generik: f(n) = g(n) + h(n).

Tidak tahu apa-apa tentang film, sehingga bisa diuji dengan graf contoh dari materi.
Pencarian tidak berhenti di goal pertama: goal dicatat lalu tetap diekspansi
sampai jumlah goal mencapai goal_limit.
"""

import heapq
import time
from collections.abc import Callable, Iterable

from recommender.results import Goal, Node, SearchResult, Step


def a_star_search(
    starts: Iterable[Node],
    neighbors: Callable[[Node], dict[Node, float]],  # tetangga -> cost
    heuristic: Callable[[Node], float],
    is_goal: Callable[[Node], bool],
    goal_limit: int,
    max_expansions: int,
) -> SearchResult:
    started_at = time.perf_counter()
    starts = list(starts)

    # Seperti materi, heapq adalah min-heap berisi (f, g, node): f terkecil keluar duluan.
    open_list: list[tuple[float, float, Node]] = []
    closed_set: set[Node] = set()
    g_score: dict[Node, float] = {}
    came_from: dict[Node, Node] = {}
    goals: list[Goal] = []
    steps: list[Step] = []

    # 1. Inisialisasi: semua start masuk Open List dengan g = 0, f = h.
    for start in starts:
        g_score[start] = 0.0
        heapq.heappush(open_list, (heuristic(start), 0.0, start))

    while open_list and len(closed_set) < max_expansions:
        # 2. Evaluasi: ambil node dengan f terkecil.
        _, g, current_node = heapq.heappop(open_list)

        # Node bisa masuk Open List lebih dari sekali; entri lama yang sudah diekspansi dilewati.
        if current_node in closed_set:
            continue

        h = heuristic(current_node)
        f = g + h

        # 3. Pengecekan goal. Closed Set menjamin setiap node hanya dicatat sekali.
        action = "start" if current_node in starts else "expand"
        if is_goal(current_node):
            action = "goal"
            goals.append(Goal(current_node, g, h, f, reconstruct_path(came_from, current_node)))

        steps.append(Step(len(steps) + 1, current_node, g, h, f, action))

        if len(goals) >= goal_limit:
            break

        # 4. Ekspansi: perbarui tetangga jika ditemukan jalur yang lebih murah.
        closed_set.add(current_node)

        for neighbor, cost in neighbors(current_node).items():
            tentative_g = g + cost

            if neighbor in closed_set or tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            g_score[neighbor] = tentative_g
            came_from[neighbor] = current_node
            heapq.heappush(open_list, (tentative_g + heuristic(neighbor), tentative_g, neighbor))

    return SearchResult(
        goals=goals,
        steps=steps,
        expanded_count=len(closed_set),
        duration_ms=(time.perf_counter() - started_at) * 1000,
    )


def reconstruct_path(came_from: dict[Node, Node], node: Node) -> list[Node]:
    """6. Rekonstruksi jalur: telusuri came_from dari goal mundur ke start."""
    path = [node]

    while node in came_from:
        node = came_from[node]
        path.append(node)

    path.reverse()
    return path
