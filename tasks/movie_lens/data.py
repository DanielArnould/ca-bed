from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

DATA_PATH = Path(__file__).with_name("movielens_curated.json")
DEFAULT_GROUP_SIZE = 4
DEFAULT_SEED = 7342


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    release_date: str | None
    video_release_date: str | None
    imdb_url: str | None
    genres: tuple[str, ...]


@dataclass(frozen=True)
class RatedMovie(Movie):
    rating: float | None
    timestamp: int | None


@dataclass(frozen=True)
class PersonaRecord:
    persona_id: int
    persona_name: str
    age: int | None
    gender: str | None
    summary: str
    preference_signature: str
    shortlist_genres: tuple[str, ...]
    favorite_motifs: tuple[str, ...]
    avoid_triggers: tuple[str, ...]
    liked_movie_ids: tuple[int, ...]
    disliked_movie_ids: tuple[int, ...]
    preferred_movie_ids: tuple[int, ...]
    preferred_movies: tuple[RatedMovie, ...]
    disliked_movies: tuple[RatedMovie, ...]
    source_user_id: int | None
    rating_count: int | None
    average_rating: float | None
    high_rating_threshold: int | None
    low_rating_threshold: int | None

    def likes_movie(self, movie_id: int) -> bool:
        return movie_id in self.preferred_movie_ids


@dataclass(frozen=True)
class PersonaMovieMatchInstance:
    personas: tuple[PersonaRecord, ...]
    target_index: int
    target_movie: RatedMovie

    @property
    def target_persona(self) -> PersonaRecord:
        return self.personas[self.target_index]


def load_dataset(
    *,
    dataset_path: Path | None = None,
    personas_per_instance: int = DEFAULT_GROUP_SIZE,
    fraction: float | None = None,
    seed: int = DEFAULT_SEED,
) -> list[PersonaMovieMatchInstance]:
    if personas_per_instance < 2:
        raise ValueError("personas_per_instance must be at least 2")

    payload = _load_payload(dataset_path or DATA_PATH)
    personas = _parse_personas(payload.get("personas", ()))

    rng = Random(seed)
    candidates = list(personas)
    rng.shuffle(candidates)

    instances = _build_instances(candidates, personas_per_instance, rng)
    rng.shuffle(instances)

    if fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in the interval (0, 1]")
        if instances:
            count = max(1, round(len(instances) * fraction))
            instances = instances[:count]
        else:
            instances = []

    return instances


def _build_instances(
    personas: Sequence[PersonaRecord],
    group_size: int,
    rng: Random,
) -> list[PersonaMovieMatchInstance]:
    results: list[PersonaMovieMatchInstance] = []

    for persona in personas:
        if not persona.preferred_movies:
            continue

        for preferred in persona.preferred_movies:
            movie_id = preferred.id
            distractors = [
                other
                for other in personas
                if other is not persona and not other.likes_movie(movie_id)
            ]
            if len(distractors) < group_size - 1:
                continue

            selected = rng.sample(distractors, group_size - 1)
            group = [persona, *selected]
            rng.shuffle(group)
            target_index = group.index(persona)

            results.append(
                PersonaMovieMatchInstance(
                    personas=tuple(group),
                    target_index=target_index,
                    target_movie=preferred,
                )
            )
            break

    return results


def _parse_personas(raw_personas: Sequence[Mapping[str, Any]]) -> tuple[PersonaRecord, ...]:
    personas: list[PersonaRecord] = []
    for idx, entry in enumerate(raw_personas):
        persona_entry = entry.get("persona", {})

        preferred_movies = tuple(
            _parse_rated_movie(movie)
            for movie in entry.get("preferred_movies", ())
        )
        disliked_movies = tuple(
            _parse_rated_movie(movie)
            for movie in entry.get("disliked_movies", ())
        )

        personas.append(
            PersonaRecord(
                persona_id=idx,
                persona_name=f"Persona-{idx}",
                age=_ensure_int(persona_entry.get("age")),
                gender=_normalise_gender(persona_entry.get("gender")),
                summary=str(persona_entry.get("summary") or ""),
                preference_signature=str(
                    persona_entry.get("preference_signature") or ""
                ),
                shortlist_genres=_ensure_str_tuple(
                    persona_entry.get("shortlist_genres", ())
                ),
                favorite_motifs=_ensure_str_tuple(
                    persona_entry.get("favorite_motifs", ())
                ),
                avoid_triggers=_ensure_str_tuple(
                    persona_entry.get("avoid_triggers", ())
                ),
                liked_movie_ids=_ensure_int_tuple(
                    persona_entry.get("liked_movie_ids", ())
                ),
                disliked_movie_ids=_ensure_int_tuple(
                    persona_entry.get("disliked_movie_ids", ())
                ),
                preferred_movie_ids=_ensure_int_tuple(
                    entry.get("preferred_movie_ids", ())
                ),
                preferred_movies=preferred_movies,
                disliked_movies=disliked_movies,
                source_user_id=_ensure_int(persona_entry.get("source_user_id")),
                rating_count=_ensure_int(persona_entry.get("rating_count")),
                average_rating=_ensure_float(persona_entry.get("average_rating")),
                high_rating_threshold=_ensure_int(
                    persona_entry.get("high_rating_threshold")
                ),
                low_rating_threshold=_ensure_int(
                    persona_entry.get("low_rating_threshold")
                ),
            )
        )

    return tuple(personas)


def _parse_rated_movie(raw_movie: Mapping[str, Any]) -> RatedMovie:
    base = _parse_movie(raw_movie)
    rating_value = raw_movie.get("rating")
    timestamp_value = raw_movie.get("timestamp")
    rating = float(rating_value) if rating_value is not None else None
    timestamp = int(timestamp_value) if timestamp_value is not None else None
    return RatedMovie(
        id=base.id,
        title=base.title,
        release_date=base.release_date,
        video_release_date=base.video_release_date,
        imdb_url=base.imdb_url,
        genres=base.genres,
        rating=rating,
        timestamp=timestamp,
    )


def _parse_movie(raw_movie: Mapping[str, Any]) -> Movie:
    return Movie(
        id=int(raw_movie["id"]),
        title=str(raw_movie.get("title") or ""),
        release_date=_ensure_str(raw_movie.get("release_date")),
        video_release_date=_ensure_str(raw_movie.get("video_release_date")),
        imdb_url=_ensure_str(raw_movie.get("imdb_url")),
        genres=_ensure_str_tuple(raw_movie.get("genres", ())),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _ensure_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _ensure_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _ensure_str_tuple(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values if value not in (None, ""))


def _ensure_int_tuple(values: Any) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(value) for value in values if value not in (None, ""))


def _normalise_gender(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text if text else None
