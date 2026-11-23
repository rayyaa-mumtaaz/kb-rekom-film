import pytest

from recommender.models import Movie
from recommender.similarity import jaccard, reason, similarity


def movie(genres=(), cast=(), director=None, keywords=()):
    return Movie(id=1, title="x", genres=list(genres), cast=list(cast), director=director, keywords=list(keywords))


def test_jaccard_for_ordinary_sets():
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5


def test_jaccard_is_zero_when_both_sets_are_empty():
    assert jaccard(set(), set()) == 0.0


def test_identical_movies_have_similarity_1():
    inception = movie(["Action", "Sci-Fi"], ["Leonardo DiCaprio"], "Christopher Nolan", ["dream"])

    assert similarity(inception, inception) == pytest.approx(1.0)


def test_weights_each_component():
    a = movie(["Action", "Drama"], ["X"], "Nolan", ["dream"])
    b = movie(["Action"], ["Y"], "Nolan", [])

    # 0.40 · 1/2 + 0.25 · 0 + 0.20 · 1 + 0.15 · 0
    assert similarity(a, b) == pytest.approx(0.40)


def test_two_null_directors_are_not_the_same_director():
    assert similarity(movie(), movie()) == 0.0


def test_reason_uses_component_with_largest_weighted_contribution():
    a = movie(["Action", "Drama", "Crime", "War"], ["A"], "Christopher Nolan", ["dream"])
    b = movie(["Action", "Comedy", "Family", "Horror"], ["B"], "Christopher Nolan", ["dream"])

    # genre 0.40 · 1/7 ≈ 0.057, sutradara 0.20, keyword 0.15 · 1 = 0.15
    assert reason(a, b) == "sutradara sama: Christopher Nolan"


def test_reason_lists_shared_members():
    a = movie(["Science Fiction", "Action", "Drama"])
    b = movie(["Science Fiction", "Action"])

    assert reason(a, b) == "genre sama: Science Fiction, Action"
