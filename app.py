"""Web lokal: hanya untuk input film dan menampilkan hasil. Jalankan: python app.py"""

from pathlib import Path

from flask import Flask, render_template, request

from recommender import config, database
from recommender.recommendation import RecommendationService
from recommender.validation import validate_titles


def create_app(database_path: Path | str = config.DATABASE_PATH) -> Flask:
    app = Flask(__name__)

    def render_home(errors: list[str] | None = None, old_titles: list[str] | None = None, status: int = 200):
        titles = database.load_titles(database.connect(database_path))
        return (
            render_template(
                "home.html",
                titles=titles,
                errors=errors or [],
                old_titles=old_titles or [],
                input_count=config.SEEDS["max"],
                required_count=config.SEEDS["min"],
            ),
            status,
        )

    @app.get("/")
    def home():
        return render_home()

    @app.post("/recommend")
    def recommend():
        connection = database.connect(database_path)
        raw_titles = request.form.getlist("movies")
        titles, errors = validate_titles(raw_titles, set(database.load_titles(connection)))

        if errors:
            return render_home(errors, raw_titles, status=422)

        movies_by_title = {movie.title: movie for movie in database.load_movies(connection).values()}
        result = RecommendationService(connection).recommend([movies_by_title[title] for title in titles])
        return render_template("result.html", result=result)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
