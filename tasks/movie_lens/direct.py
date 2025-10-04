from __future__ import annotations

import re
from textwrap import dedent
from typing import override

from node import EvidenceNode
from tasks.direct_prompting_task import DirectPromptingTask, NaiveQuestionerResponse

from .common import PersonaContext, format_candidate_movies, format_persona_context
from .data import Movie


class Direct(DirectPromptingTask):
    persona: PersonaContext
    candidate_movies: list[Movie]

    def __init__(
        self,
        max_conversation_depth: int,
        persona: PersonaContext,
        candidate_movies: list[Movie],
    ):
        if not candidate_movies:
            raise ValueError("candidate_movies must not be empty")

        movie_titles: list[str] = []
        movie_records: list[Movie] = []
        for movie in candidate_movies:
            title = str(getattr(movie, "title", "")).strip()
            if not title:
                raise ValueError("Each candidate movie must include a title")
            movie_titles.append(title)
            movie_records.append(movie)

        self.persona = persona
        self.candidate_movies = movie_records
        self.max_conversation_depth = max_conversation_depth
        self.hypothesis_space = movie_titles
        self._persona_block = format_persona_context(persona)
        self._candidate_block = format_candidate_movies(movie_records)

    def __str__(self) -> str:
        return (
            "MovieLens (Direct): "
            f"max_conversation_depth={self.max_conversation_depth} "
            f"persona={self.persona.name!r}"
        )

    @override
    def get_questioner_prompt(self, current_node: EvidenceNode) -> str:
        history: list[tuple[str, str]] = []
        node = current_node
        while node.parent:
            history.append((node.parent.question, node.answer))
            node = node.parent.parent
        history.reverse()

        base_prompt = (
            dedent("""\
            You are an insightful film curator playing a yes/no deduction game with me.

            Candidate films:
            {candidate_block}

            On each turn you may either ask a single yes/no question or make a prediction about the exact film.

            If you want to gather more information, respond with:
            [QUESTION]: <your yes/no question>

            If you are ready to identify the film, respond with:
            [PREDICTION]: <exact film title from the candidate list>

            Ask crisp questions that probe genre tone, character focus, themes, or signature moments I should be able to reply with "Yes" or "No" quickly. Avoid repeating or paraphrasing previous questions.
            """)
            .format(
                persona_block=self._persona_block,
                candidate_block=self._candidate_block,
            )
            .strip()
        )

        if not history:
            return base_prompt

        conversation_block = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        encouragement = (
            "\n\nBased on the gathered signals, start converging on a prediction."
            if len(history) >= 3
            else ""
        )
        return (
            base_prompt
            + "\n\nConversation so far:\n"
            + conversation_block
            + encouragement
        )

    @override
    def parse_questioner_output(
        self, output: str
    ) -> tuple[NaiveQuestionerResponse, str]:
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(
            r"\[(PREDICTION|ANSWER)\]:\s*(.*)", output, re.IGNORECASE
        )

        if question_match:
            return NaiveQuestionerResponse.QUESTION, question_match.group(1).strip()
        if prediction_match:
            return NaiveQuestionerResponse.PREDICTION, prediction_match.group(2).strip()
        raise RuntimeError(f"Response does not match expected structure: {output}")

    @override
    def get_answerer_prompt(self, question: str) -> str:
        persona_name = self.persona.name or "the movie lover"
        gender_note = f" {self.persona.gender.lower()}" if self.persona.gender else ""
        return (
            dedent("""\
            You are {persona_name}{gender_note}. Remain fully in character and lean on the persona briefing below when you respond.

            Persona briefing:
            {persona_block}

            Answer the curator's question exactly as this persona would. Reply with ONLY "Yes" or "No"—no additional words.
            Question: {question}
            """)
            .format(
                persona_name=persona_name,
                gender_note=gender_note,
                persona_block=self._persona_block,
                question=question,
            )
            .strip()
        )
