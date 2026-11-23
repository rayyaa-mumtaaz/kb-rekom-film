import random

from recommender.hill_climbing import hill_climbing
from recommender.results import ClimbStep


def sum_score(state):
    return float(sum(state))


def bumpy_score(state):
    # Banyak local maximum, untuk menguji random restart.
    product = 1
    for item in state:
        product *= item
    return float(product % 17)


def climb_pool(initial_state, restarts=5, seed=42, score=sum_score):
    return hill_climbing(
        pool=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        initial_state=initial_state,
        state_size=3,
        score=score,
        restarts=restarts,
        max_iterations=100,
        randomizer=random.Random(seed),
    )


def test_reaches_known_optimal_state_for_sum_score():
    result = climb_pool([1, 2, 3], restarts=1)

    assert sorted(result.best_state) == [7, 8, 9]
    assert result.best_score == 24


def test_logs_initial_state_as_iteration_0_and_one_swap_per_iteration():
    log = climb_pool([1, 2, 3], restarts=1).log[1]

    assert log[0] == ClimbStep(0, None, None, 6.0)
    assert log[1] == ClimbStep(1, 1, 9, 14.0)
    assert len(log) == 4


def test_never_lowers_score_within_a_restart():
    result = climb_pool([1, 2, 3], score=bumpy_score)

    for restart_log in result.log.values():
        scores = [step.score for step in restart_log]
        assert scores == sorted(scores)


def test_returns_best_state_across_all_restarts():
    result = climb_pool([1, 2, 3], score=bumpy_score)
    final_scores = {restart: restart_log[-1].score for restart, restart_log in result.log.items()}

    assert len(result.log) == 5
    assert result.best_score == max(final_scores.values())
    assert final_scores[result.best_restart] == result.best_score


def test_same_seed_gives_same_result():
    first = climb_pool([1, 2, 3], seed=7, score=bumpy_score)
    second = climb_pool([1, 2, 3], seed=7, score=bumpy_score)

    assert second.best_state == first.best_state
    assert second.best_restart == first.best_restart
    assert second.log == first.log
