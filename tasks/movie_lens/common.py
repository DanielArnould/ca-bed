from __future__ import annotations

import re
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import Mapping, Sequence

from .data import PersonaMovieMatchInstance, PersonaRecord, RatedMovie

QUESTIONER_ROLE_PROMPT = dedent(
    """\
    You are an insightful film curator playing a persona-matching game. Your goal is to figure out which persona in the lineup adores the target film."""
).strip()

ANSWERER_ROLE_PROMPT = dedent(
    """\
    You are roleplaying as the film lover described below. Stay true to the persona briefing when you answer."""
).strip()


@dataclass(frozen=True)
class PersonaContext:
    persona_id: int
    name: str
    age: int | None
    gender: str | None
    summary: str
    preference_signature: str
    shortlist_genres: tuple[str, ...]
    favorite_motifs: tuple[str, ...]
    avoid_triggers: tuple[str, ...]
    preferred_movies: tuple[RatedMovie, ...]
    disliked_movies: tuple[RatedMovie, ...]


@dataclass(frozen=True)
class PersonaLineup:
    personas: tuple[PersonaContext, ...]
    display_names: tuple[str, ...]
    target_index: int
    target_movie: RatedMovie

    def __post_init__(self) -> None:
        if not self.personas:
            raise ValueError("personas must not be empty")
        if len(self.personas) != len(self.display_names):
            raise ValueError("display_names must align with personas")
        if not (0 <= self.target_index < len(self.personas)):
            raise ValueError("target_index is out of range")
        object.__setattr__(
            self,
            "_name_lookup",
            {name: persona for name, persona in zip(self.display_names, self.personas)},
        )

    @property
    def target_persona(self) -> PersonaContext:
        return self.personas[self.target_index]

    @property
    def target_name(self) -> str:
        return self.display_names[self.target_index]

    @property
    def hypothesis_space(self) -> list[str]:
        return list(self.display_names)

    def persona_by_name(self, name: str) -> PersonaContext:
        try:
            return self._name_lookup[name]
        except KeyError as exc:
            raise KeyError(f"Unknown persona name: {name}") from exc


def create_persona_lineup(instance: PersonaMovieMatchInstance) -> PersonaLineup:
    personas = tuple(persona_record_to_context(record) for record in instance.personas)
    display_names = _build_unique_display_names(personas)
    return PersonaLineup(
        personas=personas,
        display_names=display_names,
        target_index=instance.target_index,
        target_movie=instance.target_movie,
    )


def persona_record_to_context(record: PersonaRecord) -> PersonaContext:
    return PersonaContext(
        persona_id=record.persona_id,
        name=record.persona_name,
        age=record.age,
        gender=record.gender,
        summary=record.summary,
        preference_signature=record.preference_signature,
        shortlist_genres=record.shortlist_genres,
        favorite_motifs=record.favorite_motifs,
        avoid_triggers=record.avoid_triggers,
        preferred_movies=record.preferred_movies,
        disliked_movies=record.disliked_movies,
    )


def build_persona_briefing(
    persona: PersonaContext,
    *,
    is_answerer: bool,
) -> str:
    sections: list[str] = []
    for builder in PERSONA_BRIEFING_BUILDERS:
        section = builder(persona, is_answerer)
        if section:
            sections.append(section)
    return "\n".join(section for section in sections if section)


def format_persona_list(
    lineup: PersonaLineup,
    *,
    is_answerer: bool,
    selected_names: Sequence[str] | None = None,
    numbered: bool = True,
) -> str:
    if selected_names is None:
        items = list(zip(lineup.display_names, lineup.personas))
    else:
        items = [(name, lineup.persona_by_name(name)) for name in selected_names]

    blocks: list[str] = []
    for idx, (name, persona) in enumerate(items, start=1):
        body = build_persona_briefing(
            persona,
            is_answerer=is_answerer,
        )
        if not body:
            continue
        header = f"{idx}. {name}" if numbered else f"- {name}"
        block = "\n".join([header, indent(body, "   ")])
        blocks.append(block.rstrip())
    return "\n\n".join(blocks)


def format_target_movie(movie: RatedMovie) -> str:
    year = _extract_year(movie.release_date)
    genres = ", ".join(movie.genres) if movie.genres else "Unknown"
    lines = [
        f"- Title: {movie.title}",
        f"- Year: {year}" if year else None,
        f"- Genres: {genres}",
    ]
    return "\n".join(line for line in lines if line)


def build_questioner_preamble(lineup: PersonaLineup) -> str:
    return dedent(
        f"""\
        {QUESTIONER_ROLE_PROMPT}

        ### Target Film
        {format_target_movie(lineup.target_movie)}

        ### Candidate Personas
        {format_persona_list(lineup, is_answerer=False)}
        """
    ).strip()


def format_history_block(history: Sequence[tuple[str, str]]) -> str | None:
    if not history:
        return None
    lines = "\n".join(f"- Q: {question}; A: {answer}" for question, answer in history)
    return dedent(
        f"""\
        These are the questions asked so far and the persona's answers:
        {lines}
        """
    ).strip()


def format_belief_block(belief_state: Mapping[str, float]) -> str | None:
    if not belief_state:
        return None
    lines = "\n".join(
        f"- {name}: {probability:.3f}"
        for name, probability in sorted(
            belief_state.items(), key=lambda item: item[1], reverse=True
        )
    )
    return dedent(
        f"""\
        Current belief over the candidate personas:
        {lines}
        """
    ).strip()


def format_possible_answers(answers: Sequence[str]) -> str:
    return "\n".join(f"- {answer}" for answer in answers)


def build_question_generation_instructions(question_count: int) -> str:
    plural = "questions" if question_count != 1 else "question"
    return dedent(
        f"""\
        ### Task
        Generate {question_count} sharp YES/NO {plural} to pose to the persona who loves the target film.
        - Each question must be answerable with exactly 'Yes' or 'No'.
        - Focus on concrete tastes, habits, or viewing preferences that would separate the personas.
        - Avoid repeating previously asked topics or paraphrasing earlier questions.

        ### Response Format
        1. <Question 1>
        2. <Question 2>
        ...
        n. <Question n>
        """
    ).strip()


def build_question_evaluation_header(
    lineup: PersonaLineup,
    question: str,
    answers: Sequence[str],
    hypotheses: Sequence[str],
) -> str:
    personas_block = format_persona_list(
        lineup,
        is_answerer=False,
        selected_names=hypotheses,
        numbered=False,
    )
    answers_block = format_possible_answers(answers)
    return dedent(
        f"""\
        {QUESTIONER_ROLE_PROMPT}

        ### Target Film
        {format_target_movie(lineup.target_movie)}

        ### Personas Under Consideration
        {personas_block}

        ### Question Under Review
        "{question}"

        ### Possible Answers
        {answers_block}
        """
    ).strip()


def build_answerer_prompt(lineup: PersonaLineup, question: str) -> str:
    persona_briefing = build_persona_briefing(
        lineup.target_persona, is_answerer=True
    )
    return dedent(
        f"""\
        {ANSWERER_ROLE_PROMPT}

        ### Persona Briefing
        {persona_briefing}

        ### Favourite Film For This Round
        {format_target_movie(lineup.target_movie)}

        ### Instructions
        - Answer the curator's question in character, relying on your briefing.
        - Reply with exactly one word: 'Yes' or 'No'. Do not add punctuation or commentary.

        ### Question
        "{question}"
        """
    ).strip()


def _build_unique_display_names(personas: Sequence[PersonaContext]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for idx, persona in enumerate(personas, start=1):
        base = persona.name or f"Persona {idx}"
        counter = seen.get(base, 0)
        if counter:
            counter += 1
            candidate = f"{base} ({counter})"
            while candidate in seen:
                counter += 1
                candidate = f"{base} ({counter})"
            seen[base] = counter
            seen[candidate] = 1
            result.append(candidate)
        else:
            seen[base] = 1
            result.append(base)
    return tuple(result)


def _build_identity_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    descriptors: list[str] = []
    if persona.age is not None and persona.age > 0:
        descriptors.append(f"{persona.age}-year-old")
    if persona.gender:
        descriptors.append(persona.gender.lower())
    descriptor = " ".join(descriptors)
    if descriptor:
        return f"Name: {persona.name or 'Unknown persona'} ({descriptor})"
    return f"Name: {persona.name or 'Unknown persona'}"


def _build_signature_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    if persona.preference_signature:
        return f"Signature: {persona.preference_signature}"
    return None


def _build_summary_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    if persona.summary:
        return f"Summary: {persona.summary}"
    return None


def _build_genres_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    return _format_section("Core genres", persona.shortlist_genres)


def _build_motifs_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    # if not is_answerer:
    #     return None
    return _format_section("Favorite motifs", persona.favorite_motifs)


def _build_avoid_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    # if not is_answerer:
    #     return None
    return _format_section("Avoids", persona.avoid_triggers)


def _build_reference_favorites_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    if not is_answerer:
        return None
    summaries = _summarise_movies(persona.preferred_movies)
    return _format_section("Reference favorites", summaries)


def _build_reference_dislikes_section(
    persona: PersonaContext, is_answerer: bool
) -> str | None:
    if not is_answerer:
        return None
    summaries = _summarise_movies(persona.disliked_movies)
    return _format_section("Reference dislikes", summaries)


PERSONA_BRIEFING_BUILDERS = (
    # _build_identity_section,
    _build_signature_section,
    _build_summary_section,
    _build_genres_section,
    _build_motifs_section,
    _build_avoid_section,
    _build_reference_favorites_section,
    _build_reference_dislikes_section,
)


def _format_section(title: str, items: Sequence[str]) -> str:
    clean_items = [item for item in items if item]
    if not clean_items:
        return f"{title}: None listed."
    bullets = "\n".join(f"- {item}" for item in clean_items)
    return f"{title}:\n{bullets}"


def _summarise_movies(movies: Sequence[RatedMovie]) -> list[str]:
    summaries: list[str] = []
    for movie in movies:
        if not movie:
            continue
        details: list[str] = []
        year = _extract_year(movie.release_date)
        if year and year not in movie.title:
            details.append(year)
        if movie.genres:
            details.append(", ".join(movie.genres))
        if movie.rating is not None:
            details.append(f"rated {movie.rating}/5")
        descriptor = movie.title
        if details:
            descriptor += f" — {' | '.join(details)}"
        summaries.append(descriptor)
    return summaries


def _extract_year(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    if match:
        return match.group(0)
    return None
