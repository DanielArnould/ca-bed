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


class Baseline(Task):
    persona: PersonaContext
    candidate_movies: list[Movie]

    def __init__(
        self,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
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
            confidence_threshold=1.0,
            hypothesis_space=movie_titles,
        )

    def __str__(self) -> str:
        return (
            "MovieLens (Non-Bayesian): "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} max_conversation_depth={self.max_conversation_depth} "
            f"persona={self.persona.name!r}"
        )

    override
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

        pruned_titles = [
            title for title, prob in current_node.belief_state.items() if prob > 1e-5
        ]
        if not pruned_titles:
            pruned_titles = list(self.hypothesis_space)

        history_block = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in history)
        belief_block = "\n".join(f"- {title}" for title in pruned_titles)

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
            Focus on distinguishing between the most plausible films:
            {belief_block}

            Devise {num_questions} incisive YES/NO questions that will best separate the remaining contenders. Ask about cinematic qualities, plot elements, themes, or emotional tone that I can answer succinctly.
            Output format:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
            """)
            .format(
                belief_block=belief_block or "- (all candidates remain possible)",
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
            You are evaluating how each candidate film would answer a YES/NO question when judged by my tastes.

            Candidate films:
            {candidate_block}

            Question:
            "{question}"

            Sort the films into two groups based on whether the truthful answer would be YES or NO if that film were my preferred pick.
            Use the exact film titles. Cover every film exactly once.

            Respond only in this format:
            Question 1: {question}
            YES: Title A, Title B, ...
            Count of YES: <integer>
            NO: Title C, Title D, ...
            Count of NO: <integer>
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
        match_yes = re.search(r"YES:\s*(.*?)\s*Count", output, re.DOTALL)
        match_no = re.search(r"NO:\s*(.*?)\s*Count", output, re.DOTALL)

        yes_items = {
            title.strip()
            for title in (match_yes.group(1).split(",") if match_yes else [])
            if title.strip() in self._movie_lookup
        }
        no_items = {
            title.strip()
            for title in (match_no.group(1).split(",") if match_no else [])
            if title.strip() in self._movie_lookup
        }

        remaining = set(self.hypothesis_space) - yes_items - no_items
        # Default remaining films to NO to avoid losing mass.
        no_items.update(remaining)

        yes_likelihoods = {
            title: (1.0 if title in yes_items else 0.0)
            for title in self.hypothesis_space
        }
        no_likelihoods = {
            title: (1.0 if title in no_items else 0.0)
            for title in self.hypothesis_space
        }

        return {"Yes": yes_likelihoods, "No": no_likelihoods}

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
