"""Feature test memakai 15 film buatan, bukan dataset asli."""

import pytest

from app import create_app
from recommender import database
from recommender.graph import build_graph
from recommender.models import Movie

GENRES = [["Action", "Thriller"], ["Drama", "Crime"], ["Science Fiction", "Adventure"]]
DIRECTORS = ["Sutradara A", "Sutradara B", "Sutradara C", "Sutradara D"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    database_path = tmp_path / "test.sqlite"
    connection = database.connect(database_path)
    movies = [
        Movie(
            id=i,
            title=f"Film {i}",
            year=2000 + i,
            vote_average=7.0,
            vote_count=1000,
            genres=GENRES[i % 3],
            cast=[f"Aktor {i}", f"Aktor {i + 1}"],
            director=DIRECTORS[i % 4],
            keywords=[f"keyword {i % 5}"],
        )
        for i in range(1, 16)
    ]
    with connection:
        database.insert_movies(connection, movies)
    build_graph(connection)
    connection.close()

    app = create_app(database_path)
    app.config["TESTING"] = True
    return app.test_client()


def post_titles(client, titles):
    return client.post("/recommend", data={"movies": titles})


def test_home_page_has_datalist_of_titles(client):
    response = client.get("/")

    assert response.status_code == 200
    assert '<datalist id="movie-titles">' in response.text
    assert "Film 7" in response.text


def test_valid_input_shows_recommendations_without_seeds(client):
    seeds = ["Film 1", "Film 2", "Film 3"]

    response = post_titles(client, [*seeds, "", ""])

    assert response.status_code == 200
    items = response.text.split('<ol class="recommendations">')[1].split("</ol>")[0]
    recommended = [line.split("</strong>")[0] for line in items.split("<strong>")[1:]]
    assert 0 < len(recommended) <= 10
    assert not set(recommended) & set(seeds)
    assert "Log A*" in response.text
    assert "Restart 1" in response.text


@pytest.mark.parametrize(
    ("titles", "message"),
    [
        (["Film 1", "Film 2", "", "", ""], "Isi minimal 3 judul film."),
        (["Film 1", "Film 2", "Tidak Ada"], "Judul &#34;Tidak Ada&#34; tidak ada di daftar film."),
        (["Film 1", "Film 1", "Film 2"], "Judul &#34;Film 1&#34; diisi lebih dari sekali."),
    ],
)
def test_invalid_input_shows_error_in_indonesian(client, titles, message):
    response = post_titles(client, titles)

    assert response.status_code == 422
    assert message in response.text
