"""Validasi input form: 3–5 judul, tidak kembar, dan harus ada di daftar film."""

from recommender import config


def validate_titles(raw_titles: list[str], known_titles: set[str]) -> tuple[list[str], list[str]]:
    """Mengembalikan (judul bersih, daftar pesan error). Input kosong dibuang lebih dulu."""
    titles = [title.strip() for title in raw_titles if title.strip()]
    errors = []

    if len(titles) < config.SEEDS["min"]:
        errors.append(f"Isi minimal {config.SEEDS['min']} judul film.")
    if len(titles) > config.SEEDS["max"]:
        errors.append(f"Isi maksimal {config.SEEDS['max']} judul film.")

    for title in sorted(set(titles), key=titles.index):
        if titles.count(title) > 1:
            errors.append(f'Judul "{title}" diisi lebih dari sekali.')
        if title not in known_titles:
            errors.append(f'Judul "{title}" tidak ada di daftar film.')

    return titles, errors
