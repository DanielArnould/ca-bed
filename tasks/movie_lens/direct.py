from __future__ import annotations

import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, get_conversation_history
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Question,
    Recommendations,
)

from .common import MovieLensInstance, format_candidate_movies, format_persona_context


class Direct(DirectPromptingTask):
    instance: MovieLensInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: MovieLensInstance,
        max_conversation_depth: int,
        recommendation_count: int = 10,
    ) -> None:
        self.instance = instance

        candidate_titles = [movie.title for movie in instance.candidate_movies]
        if not candidate_titles:
            raise ValueError("candidate_movies must not be empty")

        if recommendation_count <= 0:
            raise ValueError("recommendation_count must be positive")

        self._candidate_block = format_candidate_movies(instance.candidate_movies)
        self._persona_block = format_persona_context(instance.persona)
        self._recommendation_target = min(recommendation_count, len(candidate_titles))
        self._candidate_lookup = {title.casefold(): title for title in candidate_titles}
        self._candidate_lookup_normalised = {
            self._normalise_title(title): title for title in candidate_titles
        }

        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=None,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=candidate_titles,
        )

    def __str__(self) -> str:
        persona_name = self.instance.persona.name or "Unknown Persona"
        return (
            "MovieLens (Direct): "
            f"task_answer={self.task_answer!r} "
            f"max_conversation_depth={self.max_conversation_depth} "
            f"recommendation_count={self._recommendation_target} "
            f"persona={persona_name!r}"
        )

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Recommendations:
        parts: list[str] = []

        parts.append(
            dedent(
                f"""\
                You are an insightful film curator collaborating with a movie lover to assemble a personalised watchlist.

                ### Candidate Films
                {self._candidate_block}

                Use the user's answers to understand their taste. Ask concise YES/NO questions that reveal concrete preferences about tone, pacing, genre blends, themes, or iconic elements. Avoid repeating or paraphrasing earlier questions.
                """
            ).strip()
        )

        history = get_conversation_history(current_node)
        if history:
            formatted_history = "\n".join(
                f"- Q: {question}; A: {answer}" for question, answer in history
            )
            parts.append(
                dedent(
                    f"""
                    These are the questions asked so far and the user's answers:
                    {formatted_history}
                    """
                ).strip()
            )

        turns_remaining = max(0, self.max_conversation_depth - len(history))
        recommendation_target = self._recommendation_target

        if turns_remaining <= 1:
            parts.append(
                dedent(
                    f"""
                    You have no further opportunities to ask questions. Provide your final ranked watchlist now.

                    Respond with exactly one line in the format:
                    [RECOMMENDATIONS]: Title 1; Title 2; ...; Title {recommendation_target}

                    List {recommendation_target} distinct films from the candidate list using their EXACT titles and release dates, ordered from strongest to weakest recommendation. Include no commentary, numbering, or extra text—only the tag and semicolon-separated titles.
                    """
                ).strip()
            )
        else:
            remaining_after = turns_remaining - 1
            plural = "questions" if remaining_after != 1 else "question"
            parts.append(
                dedent(
                    f"""
                    You may ask another YES/NO question now. After this exchange you will have {remaining_after} {plural} left before submitting your final recommendations.

                    Respond with:
                    [QUESTION]: <one precise YES/NO question>
                    """
                ).strip()
            )

        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        recommendation_match = re.search(
            r"\[RECOMMENDATIONS\]:\s*(.*)", output, re.IGNORECASE | re.DOTALL
        )

        expect_recommendations = turns_remaining <= 1

        if expect_recommendations:
            if recommendation_match:
                titles = self._parse_recommendations(recommendation_match.group(1))
                return Recommendations(titles)
            raise RuntimeError(
                "Expected recommendation list but model response was not recognised: "
                f"{output}"
            )

        if recommendation_match and not expect_recommendations:
            raise RuntimeError(
                "Received recommendations before the final turn: "
                f"{output}"
            )

        if question_match:
            return Question(question_match.group(1).strip())

        raise RuntimeError(f"Response does not match expected structure: {output}")

    def _parse_recommendations(self, raw_block: str) -> tuple[str, ...]:
        entries = re.split(r"[;\n]+", raw_block)
        titles: list[str] = []
        seen: set[str] = set()

        for entry in entries:
            cleaned = entry.strip()
            if not cleaned:
                continue

            cleaned = re.sub(r"^[\u2022•\-]+\s*", "", cleaned)
            cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
            cleaned = re.sub(r"^\d+\)\s*", "", cleaned)
            cleaned = re.sub(r"^\d+\s*-\s*", "", cleaned)
            cleaned = cleaned.strip().strip('\"\'\u201c\u201d')

            if not cleaned:
                continue

            canonical = self._candidate_lookup.get(cleaned.casefold())
            if canonical is None:
                canonical = self._candidate_lookup_normalised.get(
                    self._normalise_title(cleaned)
                )

            if canonical is None or canonical in seen:
                continue

            titles.append(canonical)
            seen.add(canonical)

            if len(titles) >= self._recommendation_target:
                break

        if not titles:
            raise RuntimeError(
                "Failed to extract any valid candidate titles from recommendations: "
                f"{raw_block}"
            )

        return tuple(titles)

    @staticmethod
    def _normalise_title(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.casefold())

    @override
    async def query_answerer(self, question: str) -> str:
        persona_name = self.instance.persona.name or "the movie lover"
        gender_note = (
            f" {self.instance.persona.gender.lower()}"
            if self.instance.persona.gender
            else ""
        )

        prompt = dedent(
            f"""\
            You are {persona_name}{gender_note}. Stay fully in character and use the persona briefing below.

            Persona briefing:
            {self._persona_block}

            Answer the curator's question exactly as this persona would. Reply with ONLY "Yes" or "No"—no additional words.

            Question: {question}
            """
        ).strip()

        return await query_llm(prompt, self.answerer_session)
