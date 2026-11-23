from dataclasses import dataclass, field


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    year: int | None = None
    vote_average: float = 0.0
    vote_count: int = 0
    overview: str | None = None
    genres: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    director: str | None = None
    keywords: list[str] = field(default_factory=list)
