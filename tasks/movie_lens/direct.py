from __future__ import annotations

import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, get_conversation_history
from tasks.direct_prompting_task import DirectPromptingTask, Prediction, Question

from .common import (
    PersonaLineup,
    build_answerer_prompt,
    build_questioner_preamble,
    create_persona_lineup,
    format_history_block,
)
from .data import PersonaMovieMatchInstance


class Direct(DirectPromptingTask):
    lineup: PersonaLineup

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: PersonaMovieMatchInstance,
        max_conversation_depth: int,
    ) -> None:
        self.lineup = create_persona_lineup(instance)
        self._questioner_preamble = build_questioner_preamble(self.lineup)

        self._name_lookup = {
            name.casefold(): name for name in self.lineup.hypothesis_space
        }
        self._name_lookup_normalised = {
            self._normalise_name(name): name for name in self.lineup.hypothesis_space
        }

        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=self.lineup.target_name,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=self.lineup.hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            "MovieLens Persona Match (Direct): "
            f"answer={self.task_answer!r} "
            f"max_conversation_depth={self.max_conversation_depth}"
        )

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        history = get_conversation_history(current_node)
        turns_remaining = max(0, self.max_conversation_depth - len(history))

        parts: list[str] = [self._questioner_preamble]

        history_block = format_history_block(history)
        if history_block:
            parts.append(history_block)

        parts.append(
            dedent(
                """\
                ### Task
                Decide whether to ask another YES/NO question or to make your final prediction about which persona loves the target film.
                - Each question must be answerable with exactly 'Yes' or 'No'.
                - Focus on concrete tastes, habits, or viewing preferences that would separate the personas.
                - Avoid repeating previously asked topics or paraphrasing earlier questions.

                ### Response Format
                If you need more information, respond with:
                [QUESTION]: <one precise YES/NO question>

                If you are confident enough to guess, respond with:
                [PREDICTION]: <Persona Name>
                Use a persona name from the candidate list exactly as written."""
            ).strip()
        )

        if len(history) >= self.max_conversation_depth - 2 and turns_remaining > 1:
            parts.append(
                dedent(
                    """\
                    You are running low on turns. Consider whether you already have enough evidence to identify the correct persona."""
                ).strip()
            )

        if turns_remaining <= 1:
            parts.append(
                dedent(
                    """\
                    You have no remaining opportunities to ask questions after this response.

                    Respond with exactly one line:
                    [PREDICTION]: <Persona Name>

                    Use a persona name from the candidate list verbatim. Do not add commentary or extra text."""
                ).strip()
            )

        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        prediction_match = re.search(r"\[PREDICTION\]:", output, re.IGNORECASE)
        question_match = re.search(r"\[QUESTION\]:", output, re.IGNORECASE)

        if prediction_match:
            prediction = self._parse_prediction(output)
            return Prediction(prediction)

        if question_match:
            if turns_remaining <= 1:
                raise RuntimeError(
                    "Expected a prediction because no turns remain, but received a question."
                )
            question = self._parse_question(output)
            return Question(question)

        raise RuntimeError(
            f"Response does not match expected structure: {output}"
        )

    @override
    async def query_answerer(self, question: str) -> str:
        prompt = build_answerer_prompt(self.lineup, question)
        return await query_llm(prompt, self.answerer_session)

    def _parse_question(self, raw_block: str) -> str:
        match = re.search(r"\[QUESTION\]:\s*(.*)", raw_block, re.IGNORECASE)
        if not match:
            raise RuntimeError(
                f"Expected question marker in response but received: {raw_block}"
            )
        question = match.group(1).strip()
        if not question:
            raise RuntimeError("Question content was empty.")
        return question

    def _parse_prediction(self, raw_block: str) -> str:
        match = re.search(r"\[PREDICTION\]:\s*(.*)", raw_block, re.IGNORECASE)
        if not match:
            raise RuntimeError(
                f"Expected prediction marker in response but received: {raw_block}"
            )
        candidate = match.group(1).strip()
        if not candidate:
            raise RuntimeError("Prediction content was empty.")

        canonical = self._name_lookup.get(candidate.casefold())
        if canonical is None:
            canonical = self._name_lookup_normalised.get(
                self._normalise_name(candidate)
            )

        if canonical is None:
            raise RuntimeError(
                f"Prediction did not contain a recognisable persona: {candidate}"
            )

        return canonical

    @staticmethod
    def _normalise_name(text: str) -> str:
        cleaned = re.sub(r"[\u2018\u2019\u201c\u201d]", "", text)
        cleaned = re.sub(r"[\"'`]", "", cleaned)
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.casefold())
        return re.sub(r"\s+", " ", cleaned).strip()
