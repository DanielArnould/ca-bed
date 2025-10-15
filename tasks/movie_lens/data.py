from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence

DATA_PATH = Path(__file__).with_name("movielens_curated.json")
DEFAULT_GROUP_SIZE = 4
DEFAULT_SEED = 7342
_ARTICLES_TO_PREFIX = ("The", "An", "A")

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
            distractors: list[tuple[float, float, PersonaRecord, bool]] = []
            for other in personas:
                if other is persona or other.likes_movie(movie_id):
                    continue
                score, is_red_herring = _compute_confusion_score(
                    anchor=persona,
                    candidate=other,
                    target_movie=preferred,
                )
                jitter = rng.random() * 1e-3
                distractors.append((score, jitter, other, is_red_herring))

            if len(distractors) < group_size - 1:
                continue

            distractors.sort(key=lambda item: (item[0], item[1]), reverse=True)
            shortlist_size = max(group_size * 3, 12)
            top_candidates = distractors[:shortlist_size]

            selected_personas: list[PersonaRecord] = []
            red_herring_options = [
                candidate
                for _, _, candidate, is_red_herring in top_candidates
                if is_red_herring
            ]
            if red_herring_options:
                selected_personas.append(rng.choice(red_herring_options))

            for _, _, candidate, _ in top_candidates:
                if len(selected_personas) >= group_size - 1:
                    break
                if candidate in selected_personas:
                    continue
                selected_personas.append(candidate)

            if len(selected_personas) < group_size - 1:
                for _, _, candidate, _ in distractors:
                    if len(selected_personas) >= group_size - 1:
                        break
                    if candidate in selected_personas:
                        continue
                    selected_personas.append(candidate)

            if len(selected_personas) < group_size - 1:
                continue

            group = [persona, *selected_personas]
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


def _compute_confusion_score(
    *,
    anchor: PersonaRecord,
    candidate: PersonaRecord,
    target_movie: RatedMovie,
) -> tuple[float, bool]:
    target_genres = set(target_movie.genres or ())
    anchor_genres = set(anchor.shortlist_genres)
    candidate_genres = set(candidate.shortlist_genres)
    anchor_motifs = set(anchor.favorite_motifs)
    candidate_motifs = set(candidate.favorite_motifs)
    anchor_avoids = set(anchor.avoid_triggers)
    candidate_avoids = set(candidate.avoid_triggers)

    preferred_genres = _collect_genres(candidate.preferred_movies)
    disliked_genres = _collect_genres(candidate.disliked_movies)

    shared_genres = _jaccard_index(anchor_genres, candidate_genres)
    shared_motifs = _jaccard_index(anchor_motifs, candidate_motifs)
    shared_avoids = _jaccard_index(anchor_avoids, candidate_avoids)

    target_shortlist_alignment = _overlap_ratio(target_genres, candidate_genres)
    target_favourites_alignment = _overlap_ratio(target_genres, preferred_genres)
    target_dislikes_conflict = _overlap_ratio(target_genres, disliked_genres)
    target_avoid_conflict = _overlap_ratio(target_genres, candidate_avoids)

    similarity = (
        0.4 * shared_genres
        + 0.3 * shared_motifs
        + 0.2 * shared_avoids
        + 0.4 * target_shortlist_alignment
        + 0.3 * target_favourites_alignment
    )
    conflict = 0.4 * target_dislikes_conflict + 0.3 * target_avoid_conflict

    confusion_score = similarity + 0.5 * conflict
    is_red_herring = (
        target_movie.id in candidate.disliked_movie_ids or conflict >= 0.2
    )

    return confusion_score, is_red_herring


def _collect_genres(movies: Iterable[RatedMovie]) -> set[str]:
    genres: set[str] = set()
    for movie in movies:
        genres.update(movie.genres or ())
    return genres


def _jaccard_index(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_ratio(source: set[str], target: set[str]) -> float:
    if not source or not target:
        return 0.0
    overlap = source & target
    return len(overlap) / len(source)


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
    title = _normalise_title(raw_movie.get("title"))
    return Movie(
        id=int(raw_movie["id"]),
        title=title or "",
        release_date=_ensure_str(raw_movie.get("release_date")),
        video_release_date=_ensure_str(raw_movie.get("video_release_date")),
        imdb_url=_ensure_str(raw_movie.get("imdb_url")),
        genres=_ensure_str_tuple(raw_movie.get("genres", ())),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_title(title: str) -> str | None:
    if not title:
        return title

    title = title.strip()
    match = re.match(r"^(?P<body>.*?)(?P<year>\s*\(.*\))$", title)
    if match:
        body = match.group("body").rstrip()
        year = match.group("year").strip()
    else:
        body = title
        year = ""

    for article in _ARTICLES_TO_PREFIX:
        pattern = re.compile(rf",\s*({article})$", re.IGNORECASE)
        found = pattern.search(body)
        if not found:
            continue

        article_text = found.group(1)
        body = pattern.sub("", body).strip()
        if body:
            normalised = f"{article_text.capitalize()} {body}"
        else:
            normalised = article_text.capitalize()

        return f"{normalised} {year}".strip()

    return title


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
