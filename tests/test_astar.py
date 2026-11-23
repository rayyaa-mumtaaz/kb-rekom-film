from recommender.astar import a_star_search


def undirected_graph(edges: list[tuple[str, str, float]]) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = {}
    for a, b, cost in edges:
        graph.setdefault(a, {})[b] = float(cost)
        graph.setdefault(b, {})[a] = float(cost)
    return graph


# Graf contoh dari slide materi (edge tak berarah).
LECTURE_GRAPH = undirected_graph(
    [("A", "B", 2), ("A", "E", 3), ("B", "C", 1), ("B", "G", 9), ("E", "D", 6), ("D", "G", 1)]
)
LECTURE_HEURISTIC = {"A": 11, "B": 6, "C": 99, "E": 7, "D": 1, "G": 0}


def search_lecture_graph(is_goal, goal_limit=1, max_expansions=100):
    return a_star_search(
        starts=["A"],
        neighbors=lambda node: LECTURE_GRAPH.get(node, {}),
        heuristic=lambda node: LECTURE_HEURISTIC[node],
        is_goal=is_goal,
        goal_limit=goal_limit,
        max_expansions=max_expansions,
    )


def test_finds_lecture_path_a_e_d_g_with_cost_10():
    result = search_lecture_graph(lambda node: node == "G")

    assert len(result.goals) == 1
    assert result.goals[0].path == ["A", "E", "D", "G"]
    assert result.goals[0].g == 10
    assert result.goals[0].f == 10


def test_expands_nodes_in_order_of_smallest_f():
    result = search_lecture_graph(lambda node: node == "G")

    assert [step.node for step in result.steps] == ["A", "B", "E", "D", "G"]
    assert [step.action for step in result.steps] == ["start", "expand", "expand", "expand", "goal"]


def test_matches_a_star_example_in_materi_notebook():
    # Graf berarah dan heuristik dari materi.ipynb bagian A*.
    graph = {
        "A": {"B": 2, "C": 4},
        "B": {"D": 4, "E": 3},
        "C": {"F": 2},
        "D": {"G": 5},
        "E": {"G": 3},
        "F": {"G": 2},
        "G": {},
    }
    heuristic = {"A": 6, "B": 5, "C": 3, "D": 4, "E": 2, "F": 1, "G": 0}

    result = a_star_search(["A"], graph.get, heuristic.get, lambda node: node == "G", 1, 100)

    assert [step.node for step in result.steps] == ["A", "B", "C", "E", "F", "G"]
    assert result.goals[0].path == ["A", "B", "E", "G"]
    assert result.goals[0].g == 8


def test_stops_after_goal_limit_goals():
    result = search_lecture_graph(lambda node: node != "A", goal_limit=2)

    assert [goal.node for goal in result.goals] == ["B", "E"]


def test_respects_max_expansions():
    result = search_lecture_graph(lambda node: False, max_expansions=2)

    assert result.expanded_count == 2
    assert result.goals == []


def test_does_not_count_start_as_goal_when_is_goal_rejects_it():
    result = search_lecture_graph(lambda node: node != "A", goal_limit=10)

    assert "A" not in [goal.node for goal in result.goals]
    assert result.steps[0].action == "start"


def test_searches_from_several_starts_at_once():
    graph = {"X": {"Z": 5.0}, "Y": {"Z": 1.0}, "Z": {}}

    result = a_star_search(["X", "Y"], graph.get, lambda node: 0.0, lambda node: node == "Z", 1, 10)

    assert result.goals[0].path == ["Y", "Z"]
    assert result.goals[0].g == 1
