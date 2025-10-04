from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import override

from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from tasks.task import Task

from .common import PersonaContext, format_candidate_movies, format_persona_context
from .data import Movie


class Bayesian(Task):
    persona: PersonaContext
    candidate_movies: list[Movie]

    def __init__(
        self,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        persona: PersonaContext,
        candidate_movies: list[Movie],
    ):
        if not candidate_movies:
            raise ValueError("candidate_movies must not be empty")

        movie_titles: list[str] = []
        movie_records: list[Movie] = []
        movie_lookup: dict[str, Movie] = {}
        for movie in candidate_movies:
            title = str(getattr(movie, "title", "")).strip()
            if not title:
                raise ValueError("Each candidate movie must include a title")
            movie_titles.append(title)
            movie_records.append(movie)
            movie_lookup[title] = movie

        self.persona = persona
        self.candidate_movies = movie_records
        self._movie_lookup = movie_lookup
        self._persona_block = format_persona_context(persona)
        self._candidate_block = format_candidate_movies(movie_records)

        super().__init__(
            None,
            max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=movie_titles,
        )

    def __str__(self) -> str:
        return (
            "MovieLens (Bayesian): "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} max_conversation_depth={self.max_conversation_depth} "
            f"confidence_threshold={self.confidence_threshold} persona={self.persona.name!r}"
        )

    @override
    async def create_root(self, model: Model) -> tuple[EvidenceNode, int, int]:
        uniform_probability = 1.0 / len(self.hypothesis_space)
        uniform_prior = {hypothesis: uniform_probability for hypothesis in self.hypothesis_space}
        return EvidenceNode("ROOT", uniform_prior, 1.0), 0, 0

    @override
    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        history: list[tuple[str, str]] = []
        node = current_node
        while node.parent:
            history.append((node.parent.question, node.answer))
            node = node.parent.parent
        history.reverse()

        belief_block = "\n".join(
            f"- Movie: {title}; Probability: {prob:.4f}"
            for title, prob in current_node.belief_state.items()
        )
        history_block = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in history)

        prompt_parts = [
            dedent("""\
            You are an expert film diagnostician narrowing down which movie in a curated slate are suited for me.

            Candidate films:
            {candidate_block}
            """)
            .format(
                candidate_block=self._candidate_block,
            )
            .strip()
        ]

        if history:
            prompt_parts.append(
                dedent("""\
                Conversation history (your questions vs. my YES/NO replies):
                {history}
                """)
                .format(history=history_block)
                .strip()
            )

        prompt_parts.append(
            dedent("""\
            Current belief over the movies:
            {belief_block}

            Devise {num_questions} incisive YES/NO questions that will best separate the remaining contenders. Ask about cinematic qualities, plot elements, themes, or emotional tone that I can answer succinctly.
            Response format:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
            """)
            .format(
                belief_block=belief_block or "- (beliefs not yet established)",
                num_questions=self.max_question_nodes,
            )
            .strip()
        )

        return "\n\n".join(prompt_parts)

    @override
    def parse_question_generation_output(self, output: str) -> list[str]:
        question_texts = re.findall(r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE)
        return [text.strip().rstrip("?") + "?" for text in question_texts]

    @override
    def get_likelihood_elicitation_prompt(self, question: str) -> str:
        return (
            dedent("""\
            Estimate how likely I would answer YES to the question if each film were my preferred pick.

            Candidate films:
            {candidate_block}

            Question:
            "{question}"

            Provide a single JSON object where each key is an exact film title and each value is the probability (0.0 to 1.0) that I would answer YES if that film were a match. Cover every film exactly once.
            """)
            .format(
                candidate_block=self._candidate_block,
                question=question,
            )
            .strip()
        )

    @override
    def parse_likelihood_elicitation_output(
        self, output: str
    ) -> dict[str, dict[str, float]]:
        cleaned_output = output[output.rfind("{") : output.rfind("}") + 1]
        probs: dict[str, float] = json.loads(cleaned_output)
        filtered = {
            title: float(max(0.0, min(1.0, probs.get(title, 0.0))))
            for title in self.hypothesis_space
        }
        return {
            "Yes": filtered,
            "No": {title: 1.0 - prob for title, prob in filtered.items()},
        }

    @override
    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        persona_name = self.persona.name or "the movie lover"
        gender_note = f" {self.persona.gender.lower()}" if self.persona.gender else ""
        return (
            dedent("""\
            You are {persona_name}{gender_note}. Stay entirely in character and use the persona briefing below to ground your answer.

            Persona briefing:
            {persona_block}

            Question: {question}

            Reply with ONLY "Yes" or "No". Consider whether a movie that epitomises your tastes would truthfully earn a "Yes" to that question.
            """)
            .format(
                persona_name=persona_name,
                gender_note=gender_note,
                persona_block=self._persona_block,
                question=question_node.question,
            )
            .strip()
        )
