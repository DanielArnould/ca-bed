from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

from .data import Movie, PersonaRecord, RatedMovie


@dataclass(frozen=True)
class PersonaContext:
    name: str
    age: int | None
    gender: str | None
    summary: str
    preference_signature: str
    shortlist_genres: Sequence[str]
    favorite_motifs: Sequence[str]
    avoid_triggers: Sequence[str]
    reference_favorites: Sequence[str]
    reference_dislikes: Sequence[str]


def persona_record_to_context(
    persona: PersonaRecord,
    preferred_movies: Sequence[RatedMovie],
    disliked_movies: Sequence[RatedMovie],
) -> PersonaContext:
    return PersonaContext(
        name=persona.persona_name,
        age=persona.age,
        gender=persona.gender,
        summary=persona.summary,
        preference_signature=persona.preference_signature,
        shortlist_genres=persona.shortlist_genres,
        favorite_motifs=persona.favorite_motifs,
        avoid_triggers=persona.avoid_triggers,
        reference_favorites=_summarise_movies(preferred_movies),
        reference_dislikes=_summarise_movies(disliked_movies),
    )


def format_persona_context(persona: PersonaContext) -> str:
    lines: list[str] = []

    if persona.name:
        traits: list[str] = []
        # if persona.age is not None:
        #     traits.append(f"{persona.age}-year-old")
        if persona.gender:
            traits.append(persona.gender.lower())
        descriptor = " ".join(traits)
        if descriptor:
            lines.append(f"Name: {persona.name} ({descriptor})")
        else:
            lines.append(f"Name: {persona.name}")
    else:
        lines.append("Name: Unknown persona")

    if persona.preference_signature:
        lines.append(f"Signature: {persona.preference_signature}")
    if persona.summary:
        lines.append(f"Summary: {persona.summary}")

    lines.append(_format_section("Core genres", persona.shortlist_genres))
    lines.append(_format_section("Favorite motifs", persona.favorite_motifs))
    lines.append(_format_section("Avoids", persona.avoid_triggers))
    lines.append(_format_section("Reference favorites", persona.reference_favorites))
    lines.append(_format_section("Reference dislikes", persona.reference_dislikes))

    return "\n".join(item for item in lines if item)


MovieLike = Union[Movie, RatedMovie, Mapping[str, Any]]


def format_candidate_movies(movies: Sequence[MovieLike]) -> str:
    if not movies:
        return "No candidate films supplied."

    formatted: list[str] = []
    for idx, movie in enumerate(movies, start=1):
        title = str(_movie_field(movie, "title", f"Movie {idx}"))
        descriptor = f"{idx}. {title}"
        formatted.append(descriptor)

    return "\n".join(formatted)


@dataclass(frozen=True)
class MovieLensInstance:
    persona: PersonaContext
    candidate_movies: tuple[Movie, ...]
    target_movie: Movie | None
    preferred_movies: tuple[RatedMovie, ...]
    disliked_movies: tuple[RatedMovie, ...]

    def __init__(
        self,
        persona: PersonaContext,
        candidate_movies: Sequence[Movie],
        target_movie: Movie | None = None,
        preferred_movies: Sequence[RatedMovie] | None = None,
        disliked_movies: Sequence[RatedMovie] | None = None,
    ) -> None:
        if not candidate_movies:
            raise ValueError("candidate_movies must not be empty")

        normalised_candidates = tuple(candidate_movies)
        candidate_titles = {movie.title for movie in normalised_candidates}

        if target_movie is not None and target_movie.title not in candidate_titles:
            raise ValueError(
                "target_movie must be included in candidate_movies"
            )

        object.__setattr__(self, "persona", persona)
        object.__setattr__(self, "candidate_movies", normalised_candidates)
        object.__setattr__(self, "target_movie", target_movie)
        object.__setattr__(self, "preferred_movies", tuple(preferred_movies or ()))
        object.__setattr__(self, "disliked_movies", tuple(disliked_movies or ()))

    @property
    def hypothesis_space(self) -> list[str]:
        return [movie.title for movie in self.candidate_movies]


def _format_section(title: str, items: Sequence[str]) -> str:
    clean_items = [item for item in items if item]
    if not clean_items:
        return f"{title}: None listed."
    bullets = "\n".join(f"- {item}" for item in clean_items)
    return f"{title}:\n{bullets}"


def _extract_year(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return match.group(0)
    return None


def _summarise_movies(movies: Sequence[RatedMovie]) -> list[str]:
    summaries: list[str] = []
    for movie in movies:
        title = movie.title
        details: list[str] = []
        year = _extract_year(movie.release_date)
        if year and year not in title:
            details.append(year)
        if movie.genres:
            details.append(", ".join(movie.genres))
        if movie.rating is not None:
            details.append(f"rated {movie.rating}/5")
        descriptor = title
        if details:
            descriptor += f" — {' | '.join(details)}"
        summaries.append(descriptor)
    return summaries


def _movie_field(movie: MovieLike, field: str, default: Any = None) -> Any:
    if hasattr(movie, field):
        return getattr(movie, field)
    if isinstance(movie, Mapping):
        return movie.get(field, default)
    return default
