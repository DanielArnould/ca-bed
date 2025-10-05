from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from models import LLMRequestSession, llm_models, query_llm

if TYPE_CHECKING:
    from .common import MovieLensInstance, PersonaContext

LOGGER = logging.getLogger("MovieLensPersonaPipeline")

API_CALL_DELAY_SECONDS = 0.5

PERSONA_PROMPT_TEMPLATE = """You are an imaginative researcher crafting nuanced movie-goer personas.
Work from real MovieLens ratings to produce a persona that feels human and evocative.
Real user context:
- User ID: {user_id}
- Total ratings: {rating_count}
- Average rating: {average_rating}
- Dominant genres: {top_genres}
- Age: {user_age}
- Gender: {user_gender}

Movies this user rated {high_threshold}/5 or higher:
{liked_movies}

Movies this user rated {low_threshold}/5 or lower:
{disliked_movies}

Describe the persona indirectly—lean on sensory memories, themes, pacing preferences, and viewing habits rather than explicit genre lists.
Stay aligned with the provided age and gender; do not invent conflicting details.
Return STRICT JSON with this schema:
{{
  "persona_name": <creative full name>,
  "age": {age_instruction},
  "summary": <two vivid sentences>,
  "preference_signature": <concise 3-6 word vibe>,
  "shortlist_genres": [<use the dominant genres above>],
  "favorite_motifs": [<at least three sensory or thematic motifs>],
  "avoid_triggers": [<at least two elements this persona avoids>]
}}
Do not add commentary or code fences—JSON only."""


HIGH_RATING_THRESHOLD = 4
LOW_RATING_THRESHOLD = 2
MIN_HIGH_RATED_MOVIES = 5
PROMPT_LIKES_LIMIT = 12
PROMPT_DISLIKES_LIMIT = 6


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    release_date: str | None
    video_release_date: str | None
    imdb_url: str | None
    genres: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "release_date": self.release_date,
            "video_release_date": self.video_release_date,
            "imdb_url": self.imdb_url,
            "genres": self.genres,
        }
    
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RatedMovie":
        return cls(
            id=payload.get("id"),
            title=payload.get("title"),
            release_date=payload.get("release_date"),
            video_release_date=payload.get("video_release_date"),
            imdb_url=payload.get("imdb_url"),
            genres=list(payload.get("genres", []))
        )


@dataclass(frozen=True)
class MovieRating:
    movie: Movie
    rating: int
    timestamp: int


@dataclass(frozen=True)
class UserDemographics:
    age: int | None
    gender: str | None


@dataclass
class UserProfile:
    user_id: int
    age: int | None
    gender: str | None
    ratings: list[MovieRating]
    high_rated: list[MovieRating]
    low_rated: list[MovieRating]
    average_rating: float

    @property
    def rating_count(self) -> int:
        return len(self.ratings)



@dataclass(frozen=True)
class MovieLensPaths:
    data_dir: Path
    users: Path
    movies: Path
    ratings: Path
    genres: Path
    occupations: Path

    @classmethod
    def build(cls, data_dir: Path) -> "MovieLensPaths":
        data_dir = data_dir.expanduser().resolve()
        return cls(
            data_dir=data_dir,
            users=data_dir / "u.user",
            movies=data_dir / "u.item",
            ratings=data_dir / "u.data",
            genres=data_dir / "u.genre",
            occupations=data_dir / "u.occupation",
        )


def _read_lines(path: Path) -> Iterable[str]:
    with path.open(encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield line


def load_genre_catalog(path: Path) -> list[str]:
    mapping: dict[int, str] = {}
    for line in _read_lines(path):
        name, idx = line.split("|")
        mapping[int(idx)] = name
    return [name for _, name in sorted(mapping.items())]


def load_movies(path: Path, genre_catalog: list[str]) -> list[Movie]:
    movies: list[Movie] = []
    for line in _read_lines(path):
        parts = line.split("|")
        if len(parts) < 5:
            raise ValueError(f"Malformed movie line: {line!r}")
        movie_id = int(parts[0])
        title = parts[1]
        release_date = parts[2] or None
        video_release_date = parts[3] or None
        imdb_url = parts[4] or None
        flags = parts[5:]
        genres = [genre_catalog[idx] for idx, flag in enumerate(flags) if flag == "1" and idx < len(genre_catalog)]
        movies.append(
            Movie(
                id=movie_id,
                title=title,
                release_date=release_date,
                video_release_date=video_release_date,
                imdb_url=imdb_url,
                genres=genres,
            )
        )
    movies.sort(key=lambda movie: movie.id)
    return movies


def build_movie_lookup(movies: list[Movie]) -> dict[int, Movie]:
    return {movie.id: movie for movie in movies}


class RatingError(RuntimeError):
    pass


def load_user_demographics(path: Path) -> dict[int, UserDemographics]:
    demographics: dict[int, UserDemographics] = {}
    for line in _read_lines(path):
        parts = line.split("|")
        if len(parts) < 3:
            raise ValueError(f"Malformed user line: {line!r}")
        user_id_str, age_str, gender_str, *_rest = parts
        user_id = int(user_id_str)

        age: int | None
        age_str = age_str.strip()
        if age_str:
            try:
                age = int(age_str)
            except ValueError as exc:
                raise ValueError(f"Invalid age value for user {user_id}: {age_str!r}") from exc
        else:
            age = None

        gender_str = gender_str.strip()
        if gender_str:
            gender_lookup = {"M": "Male", "F": "Female"}
            gender = gender_lookup.get(gender_str.upper(), gender_str)
        else:
            gender = None

        demographics[user_id] = UserDemographics(age=age, gender=gender)
    return demographics


def load_ratings_by_user(path: Path) -> dict[int, list[tuple[int, int, int]]]:
    ratings_by_user: dict[int, list[tuple[int, int, int]]] = {}
    for line in _read_lines(path):
        try:
            user_id_str, movie_id_str, rating_str, ts_str = line.split('	')
        except ValueError as exc:
            raise RatingError(f"Malformed rating line: {line!r}") from exc
        user_id = int(user_id_str)
        movie_id = int(movie_id_str)
        rating = int(rating_str)
        timestamp = int(ts_str)
        ratings_by_user.setdefault(user_id, []).append((movie_id, rating, timestamp))
    return ratings_by_user


def build_user_profiles(
    ratings_by_user: dict[int, list[tuple[int, int, int]]],
    movie_lookup: dict[int, Movie],
    min_high_rated: int,
    user_demographics: dict[int, UserDemographics] | None = None,
    high_rating_threshold: int = HIGH_RATING_THRESHOLD,
    low_rating_threshold: int = LOW_RATING_THRESHOLD,
) -> list[UserProfile]:
    profiles: list[UserProfile] = []
    for user_id, entries in ratings_by_user.items():
        ratings: list[MovieRating] = []
        high: list[MovieRating] = []
        low: list[MovieRating] = []
        total_score = 0
        for movie_id, score, timestamp in entries:
            movie = movie_lookup.get(movie_id)
            if movie is None:
                continue
            record = MovieRating(movie=movie, rating=score, timestamp=timestamp)
            ratings.append(record)
            total_score += score
            if score >= high_rating_threshold:
                high.append(record)
            elif score <= low_rating_threshold:
                low.append(record)
        if len(high) < min_high_rated:
            continue
        ratings.sort(key=lambda r: (-r.rating, r.timestamp, r.movie.id))
        high.sort(key=lambda r: (-r.rating, r.timestamp, r.movie.id))
        low.sort(key=lambda r: (r.rating, r.timestamp, r.movie.id))
        average_rating = total_score / len(ratings) if ratings else 0.0
        if user_demographics is not None:
            demo = user_demographics.get(user_id)
            profile_age = demo.age if demo else None
            profile_gender = demo.gender if demo else None
        else:
            profile_age = None
            profile_gender = None
        profiles.append(
            UserProfile(
                user_id=user_id,
                age=profile_age,
                gender=profile_gender,
                ratings=ratings,
                high_rated=high,
                low_rated=low,
                average_rating=average_rating,
            )
        )
    profiles.sort(key=lambda profile: profile.user_id)
    return profiles




def build_movies_by_genre(movies: list[Movie]) -> dict[str, list[Movie]]:
    movies_by_genre: dict[str, list[Movie]] = {}
    for movie in movies:
        for genre in movie.genres:
            movies_by_genre.setdefault(genre, []).append(movie)
    # ensure deterministic order per genre
    for genre_movies in movies_by_genre.values():
        genre_movies.sort(key=lambda movie: movie.id)
    return movies_by_genre


def build_balanced_hypothesis_pool(
    movies_by_genre: dict[str, list[Movie]],
    genre_catalog: list[str],
    pool_size: int,
    rng: random.Random,
    required_movies: Iterable[Movie] | None = None,
) -> list[Movie]:
    if pool_size < len(genre_catalog):
        raise ValueError(
            f"pool_size={pool_size} must be at least the number of genres ({len(genre_catalog)})"
        )

    selected: list[Movie] = []
    used_ids: set[int] = set()
    genre_counts: Counter[str] = Counter()
    available_movie_ids = {movie.id for movies in movies_by_genre.values() for movie in movies}

    if required_movies is not None:
        for movie in required_movies:
            if movie.id in used_ids or movie.id not in available_movie_ids:
                continue
            selected.append(movie)
            used_ids.add(movie.id)
            for genre in movie.genres:
                genre_counts[genre] += 1

    remaining_slots = pool_size - len(selected)
    if remaining_slots <= 0:
        selected.sort(key=lambda movie: movie.id)
        return selected[:pool_size]

    per_genre = remaining_slots // len(genre_catalog)
    remainder = remaining_slots % len(genre_catalog)

    for idx, genre in enumerate(genre_catalog):
        pool = [movie for movie in movies_by_genre.get(genre, []) if movie.id not in used_ids]
        if not pool:
            continue
        target = per_genre + (1 if idx < remainder else 0)
        existing = genre_counts.get(genre, 0)
        target = max(0, target - existing)
        if target <= 0:
            continue
        sample_size = min(target, len(pool))
        choices = rng.sample(pool, sample_size) if sample_size < len(pool) else pool[:sample_size]
        for movie in choices:
            if movie.id in used_ids:
                continue
            selected.append(movie)
            used_ids.add(movie.id)
            for movie_genre in movie.genres:
                genre_counts[movie_genre] += 1

    if len(selected) < pool_size:
        for genre in genre_catalog:
            for movie in movies_by_genre.get(genre, []):
                if movie.id in used_ids:
                    continue
                selected.append(movie)
                used_ids.add(movie.id)
                if len(selected) == pool_size:
                    break
            if len(selected) == pool_size:
                break

    selected.sort(key=lambda movie: movie.id)
    return selected[:pool_size]


def compute_top_genres(ratings: list[MovieRating], limit: int = 4) -> list[str]:
    if not ratings:
        return []
    counter: Counter[str] = Counter()
    for record in ratings:
        for genre in record.movie.genres:
            counter[genre] += 1
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [genre for genre, _ in ordered[:limit]]


def format_ratings_for_prompt(ratings: list[MovieRating], limit: int) -> str:
    if not ratings:
        return 'None'
    lines = []
    for record in ratings[:limit]:
        genres = ', '.join(record.movie.genres) if record.movie.genres else 'Unknown genre'
        lines.append(
            f"- {record.movie.title} | Rating {record.rating}/5 | Genres: {genres}"
        )
    return "\n".join(lines)


def movie_rating_to_payload(record: MovieRating) -> dict[str, Any]:
    payload = record.movie.to_payload()
    payload.update({"rating": record.rating, "timestamp": record.timestamp})
    return payload




def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s][%(levelname)s] %(message)s")


def build_genre_lookup(genre_catalog: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for genre in genre_catalog:
        lowered = genre.lower()
        lookup[lowered] = genre
        lookup[lowered.replace("'", "")] = genre
        lookup[lowered.replace("-", "")] = genre
        lookup[lowered.replace(" ", "")] = genre
    return lookup


def normalise_genre(raw: str, lookup: dict[str, str]) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    direct = lookup.get(key)
    if direct:
        return direct
    squashed = key.replace("'", "").replace("-", "").replace(" ", "")
    if squashed in lookup:
        return lookup[squashed]
    matches = get_close_matches(key, lookup.keys(), n=1, cutoff=0.8)
    if matches:
        return lookup[matches[0]]
    return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.strip("`")
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```", 2)
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            return body.strip()
    return stripped


def build_persona_prompt(
    profile: UserProfile,
    top_genres: list[str],
    liked_movies_block: str,
    disliked_movies_block: str,
) -> str:
    top_genres_text = ", ".join(top_genres) if top_genres else "None"
    if profile.age is not None:
        age_text = str(profile.age)
        age_instruction = f"<integer; use the provided age of {age_text}>"
    else:
        age_text = "Unknown"
        age_instruction = "<integer>"
    gender_text = profile.gender if profile.gender is not None else "Unknown"
    return PERSONA_PROMPT_TEMPLATE.format(
        user_id=profile.user_id,
        rating_count=profile.rating_count,
        average_rating=f"{profile.average_rating:.2f}",
        top_genres=top_genres_text,
        user_age=age_text,
        user_gender=gender_text,
        age_instruction=age_instruction,
        liked_movies=liked_movies_block if liked_movies_block else "None provided",
        disliked_movies=disliked_movies_block if disliked_movies_block else "None provided",
        high_threshold=HIGH_RATING_THRESHOLD,
        low_threshold=LOW_RATING_THRESHOLD,
    )


async def generate_persona(
    profile: UserProfile,
    model_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], int, int, list[str]]:
    top_genres = compute_top_genres(profile.high_rated)
    liked_block = format_ratings_for_prompt(profile.high_rated, PROMPT_LIKES_LIMIT)
    disliked_block = format_ratings_for_prompt(profile.low_rated, PROMPT_DISLIKES_LIMIT)
    prompt = build_persona_prompt(profile, top_genres, liked_block, disliked_block)
    session = LLMRequestSession(model_key)
    async with semaphore:
        response_text = await query_llm(prompt, session)
        await asyncio.sleep(API_CALL_DELAY_SECONDS)
    prompt_tokens = session.total_input_tokens
    completion_tokens = session.total_output_tokens
    payload_text = strip_code_fence(response_text)
    try:
        persona = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Persona for user {profile.user_id} response was not valid JSON: {payload_text!r}"
        ) from exc
    if profile.age is not None:
        persona["age"] = profile.age
    else:
        persona.setdefault("age", None)
    if profile.gender is not None:
        persona["gender"] = profile.gender
    else:
        persona.setdefault("gender", None)
    persona["shortlist_genres"] = top_genres
    persona["source_user_id"] = profile.user_id
    persona["rating_count"] = profile.rating_count
    persona["average_rating"] = round(profile.average_rating, 3)
    persona["high_rating_threshold"] = HIGH_RATING_THRESHOLD
    persona["low_rating_threshold"] = LOW_RATING_THRESHOLD
    persona["liked_movie_ids"] = [record.movie.id for record in profile.high_rated]
    persona["disliked_movie_ids"] = [record.movie.id for record in profile.low_rated]
    return persona, prompt_tokens, completion_tokens, top_genres


def validate_persona(persona: dict[str, Any], genre_catalog: list[str]) -> dict[str, Any]:
    required_fields = [
        "persona_name",
        "age",
        "summary",
        "preference_signature",
        "shortlist_genres",
        "favorite_motifs",
        "avoid_triggers",
    ]
    for field in required_fields:
        if field not in persona:
            raise ValueError(f"Persona missing required field '{field}': {persona}")
    if not isinstance(persona["shortlist_genres"], list) or len(persona["shortlist_genres"]) == 0:
        raise ValueError(f"Persona shortlist_genres must be a non-empty list: {persona}")

    lookup = build_genre_lookup(genre_catalog)
    normalised: list[str] = []
    for raw_genre in persona["shortlist_genres"]:
        canonical = normalise_genre(str(raw_genre), lookup)
        if canonical and canonical not in normalised:
            normalised.append(canonical)
    if not normalised:
        raise ValueError(f"Persona shortlist_genres could not be matched to catalog: {persona}")
    persona["shortlist_genres"] = normalised
    return persona


async def build_curated_dataset(
    persona_count: int,
    paths: MovieLensPaths,
    model_key: str,
    seed: int,
    max_concurrency: int,
    pool_size: int | None,
) -> dict[str, Any]:
    genre_catalog = load_genre_catalog(paths.genres)
    movies = load_movies(paths.movies, genre_catalog)
    movie_lookup = build_movie_lookup(movies)
    user_demographics = load_user_demographics(paths.users)
    ratings_by_user = load_ratings_by_user(paths.ratings)
    profiles = build_user_profiles(
        ratings_by_user,
        movie_lookup,
        user_demographics=user_demographics,
        min_high_rated=MIN_HIGH_RATED_MOVIES,
    )

    if not profiles:
        raise RuntimeError(
            "No user profiles found with sufficient high-rated movies to build personas."
        )
    if persona_count > len(profiles):
        raise ValueError(
            f"Requested {persona_count} personas but only {len(profiles)} user profiles are available."
        )

    movies_by_genre = build_movies_by_genre(movies)

    if pool_size is None:
        pool_size = len(movies)
    requested_pool_size = pool_size

    if pool_size < len(genre_catalog):
        raise ValueError(
            f"pool_size={pool_size} must be at least the number of genres ({len(genre_catalog)})"
        )

    if pool_size > len(movies):
        LOGGER.warning(
            "Requested pool_size=%s exceeds available movies (%s); capping to catalog size",
            pool_size,
            len(movies),
        )
        pool_size = len(movies)

    rng = random.Random(seed)
    if persona_count < len(profiles):
        selected_profiles = rng.sample(profiles, persona_count)
    else:
        selected_profiles = profiles.copy()

    semaphore = asyncio.Semaphore(max_concurrency)
    persona_results = await asyncio.gather(
        *[generate_persona(profile, model_key, semaphore) for profile in selected_profiles]
    )

    hypothesis_pool = build_balanced_hypothesis_pool(
        movies_by_genre=movies_by_genre,
        genre_catalog=genre_catalog,
        pool_size=pool_size,
        rng=rng,
        required_movies=None,
    )

    curated_entries: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for profile, (persona, prompt_tokens, completion_tokens, _top_genres) in zip(
        selected_profiles, persona_results
    ):
        validated = validate_persona(persona, genre_catalog)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        favorites = profile.high_rated
        disliked = profile.low_rated

        curated_entries.append(
            {
                "persona": validated,
                "preferred_movie_ids": [record.movie.id for record in favorites],
                "preferred_movies": [movie_rating_to_payload(record) for record in favorites],
                "disliked_movie_ids": [record.movie.id for record in disliked],
                "disliked_movies": [movie_rating_to_payload(record) for record in disliked],
            }
        )

    curated_entries.sort(key=lambda entry: entry["persona"]["persona_name"])
    hypothesis_pool_payload = [movie.to_payload() for movie in hypothesis_pool]

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(paths.data_dir),
        "persona_count": persona_count,
        "requested_hypothesis_pool_size": requested_pool_size,
        "hypothesis_pool_size": len(hypothesis_pool),
        "model": model_key,
        "seed": seed,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "min_high_rated_movies": MIN_HIGH_RATED_MOVIES,
        "high_rating_threshold": HIGH_RATING_THRESHOLD,
        "low_rating_threshold": LOW_RATING_THRESHOLD,
    }

    return {
        "metadata": metadata,
        "genre_catalog": genre_catalog,
        "hypothesis_pool": hypothesis_pool_payload,
        "personas": curated_entries,
    }


def detect_default_data_dir() -> Path | None:
    repo_dir = Path(__file__).resolve().parents[2]
    candidate = repo_dir / "tasks" / "movie_lens"
    if candidate.exists():
        return candidate
    candidate = repo_dir / "ml-100k"
    if candidate.exists():
        return candidate
    cwd = Path.cwd()
    expected = {"u.user", "u.item", "u.data", "u.genre", "u.occupation"}
    if cwd.exists() and expected.issubset({item.name for item in cwd.iterdir()}):
        return cwd
    return None


def parse_args() -> argparse.Namespace:
    default_data_dir = detect_default_data_dir()
    parser = argparse.ArgumentParser(description="Curate a MovieLens persona dataset")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Directory containing raw MovieLens 100K files")
    parser.add_argument("--output", type=Path, default=None, help="Path to write movielens_curated.json (defaults to <data-dir>/movielens_curated.json)")
    parser.add_argument("--count", type=int, default=5, help="Number of personas to generate")
    parser.add_argument("--pool-size", type=int, default=None, help="Total number of movies in the combined hypothesis pool (defaults to entire catalog when omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for movie selection")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt_5",
        help="Model key from models.llm_models to use for persona generation",
    )
    parser.add_argument("--max-concurrency", type=int, default=3, help="Maximum simultaneous LLM calls")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.data_dir is None:
        raise SystemExit("--data-dir is required when automatic detection fails")
    if args.output is None:
        args.output = args.data_dir / "movielens_curated.json"
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.pool_size is not None and args.pool_size <= 0:
        raise SystemExit("--pool-size must be positive when provided")
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")
    return args


def resolve_model(model_name: str) -> str:
    if model_name not in llm_models:
        available = ", ".join(sorted(llm_models.keys()))
        raise SystemExit(
            f"Unknown model '{model_name}'. Available options: {available}"
        )
    return model_name

@dataclass(frozen=True)
class PersonaRecord:
    persona_name: str
    age: int | None
    gender: str | None
    summary: str
    preference_signature: str
    shortlist_genres: list[str]
    favorite_motifs: list[str]
    avoid_triggers: list[str]
    source_user_id: int | None = None
    rating_count: int | None = None
    average_rating: float | None = None
    high_rating_threshold: int | None = None
    low_rating_threshold: int | None = None
    liked_movie_ids: list[int] | None = None
    disliked_movie_ids: list[int] | None = None


@dataclass(frozen=True)
class RatedMovie(Movie):
    rating: int | None = None
    timestamp: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if self.rating is not None:
            payload["rating"] = self.rating
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RatedMovie":
        return cls(
            id=payload.get("id"),
            title=payload.get("title"),
            release_date=payload.get("release_date"),
            video_release_date=payload.get("video_release_date"),
            imdb_url=payload.get("imdb_url"),
            genres=list(payload.get("genres", [])),
            rating=payload.get("rating"),
            timestamp=payload.get("timestamp"),
        )


def _persona_from_payload(payload: dict[str, Any]) -> PersonaRecord:
    return PersonaRecord(
        persona_name=payload.get("persona_name", "Unknown"),
        age=payload.get("age"),
        gender=payload.get("gender"),
        summary=payload.get("summary", ""),
        preference_signature=payload.get("preference_signature", ""),
        shortlist_genres=list(payload.get("shortlist_genres", [])),
        favorite_motifs=list(payload.get("favorite_motifs", [])),
        avoid_triggers=list(payload.get("avoid_triggers", [])),
        source_user_id=payload.get("source_user_id"),
        rating_count=payload.get("rating_count"),
        average_rating=payload.get("average_rating"),
        high_rating_threshold=payload.get("high_rating_threshold"),
        low_rating_threshold=payload.get("low_rating_threshold"),
        liked_movie_ids=list(payload.get("liked_movie_ids", []) or []),
        disliked_movie_ids=list(payload.get("disliked_movie_ids", []) or []),
    )


def load_balanced_dataset(
    dataset_path: Path | None = None,
    fraction: float = 1.0,
    seed: int | None = 42,
) -> list["MovieLensInstance"]:
    if dataset_path is None:
        default_dir = detect_default_data_dir()
        if default_dir is None:
            raise FileNotFoundError(
                "Unable to infer MovieLens dataset directory; please provide dataset_path"
            )
        dataset_path = Path(default_dir) / "movielens_curated.json"

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in the interval (0, 1]")

    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Curated dataset not found at {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    hypothesis_pool = [Movie.from_payload(movie) for movie in data.get("hypothesis_pool", [])]
    personas = data.get("personas", [])

    from .common import (
        MovieLensInstance,
        PersonaContext,
        persona_record_to_context,
    )

    persona_entries: list[
        tuple[PersonaContext, tuple[RatedMovie, ...], tuple[RatedMovie, ...], frozenset[int]]
    ] = []

    for entry in personas:
        persona_payload = entry.get("persona", {})
        preferred_payloads = entry.get("preferred_movies", [])
        disliked_payloads = entry.get("disliked_movies", [])

        persona_record = _persona_from_payload(persona_payload)
        preferred_movies = tuple(
            RatedMovie.from_payload(p) for p in preferred_payloads
        )
        disliked_movies = tuple(
            RatedMovie.from_payload(p) for p in disliked_payloads
        )

        persona_context = persona_record_to_context(
            persona_record, preferred_movies, disliked_movies
        )

        preferred_ids = frozenset(
            movie.id for movie in preferred_movies if getattr(movie, "id", None) is not None
        )

        persona_entries.append(
            (persona_context, preferred_movies, disliked_movies, preferred_ids)
        )

    persona_entries.sort(key=lambda item: item[0].name or "")

    if fraction >= 1 or not persona_entries:
        selected_entries = persona_entries
    else:
        rng = random.Random(seed)
        target_size = max(1, int(round(len(persona_entries) * fraction)))

        grouped: dict[
            frozenset[int],
            list[
                tuple[
                    PersonaContext,
                    tuple[RatedMovie, ...],
                    tuple[RatedMovie, ...],
                    frozenset[int],
                ]
            ],
        ] = {}
        for entry in persona_entries:
            grouped.setdefault(entry[3], []).append(entry)

        sampled_entries: list[
            tuple[
                PersonaContext,
                tuple[RatedMovie, ...],
                tuple[RatedMovie, ...],
                frozenset[int],
            ]
        ] = []

        for group_entries in grouped.values():
            group_size = len(group_entries)
            desired = max(1, int(round(group_size * fraction)))
            if desired >= group_size:
                sampled_entries.extend(group_entries)
            else:
                sampled_entries.extend(rng.sample(group_entries, desired))

        if len(sampled_entries) > target_size:
            sampled_entries = rng.sample(sampled_entries, target_size)
        elif len(sampled_entries) < target_size:
            remaining = [entry for entry in persona_entries if entry not in sampled_entries]
            needed = target_size - len(sampled_entries)
            sampled_entries.extend(rng.sample(remaining, min(needed, len(remaining))))

        sampled_entries.sort(key=lambda item: item[0].name or "")
        selected_entries = sampled_entries

    instances = [
        MovieLensInstance(
            persona=context,
            candidate_movies=hypothesis_pool,
            target_movie=None,
            preferred_movies=preferred,
            disliked_movies=disliked,
        )
        for context, preferred, disliked, _ in selected_entries
    ]

    return instances





async def async_main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    paths = MovieLensPaths.build(args.data_dir)
    model_key = resolve_model(args.model)
    pool_size = args.pool_size

    dataset = await build_curated_dataset(
        persona_count=args.count,
        paths=paths,
        model_key=model_key,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        pool_size=pool_size,
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote curated dataset to %s", output_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    # records, hyp_set = load_balanced_dataset()
    # print(hyp_set)
    main()
