from __future__ import annotations

from textwrap import dedent
from typing import override

from tenacity import retry, stop_after_attempt

from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import (
    Task,
    parse_answer,
    parse_binary_questions,
    parse_probabilities,
)

from .common import MovieLensInstance, format_candidate_movies, format_persona_context


class Bayesian(Task):
    instance: MovieLensInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: MovieLensInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
    ) -> None:
        self.instance = instance
        self._persona_block = format_persona_context(instance.persona)
        self._candidate_block = format_candidate_movies(instance.candidate_movies)

        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=instance.target_movie.title if instance.target_movie else None,
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=instance.hypothesis_space,
        )

    def __str__(self) -> str:
        persona_name = self.instance.persona.name or "Unknown Persona"
        return (
            "MovieLens (Bayesian): "
            f"task_answer={self.task_answer!r} "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} "
            f"max_conversation_depth={self.max_conversation_depth} "
            f"confidence_threshold={self.confidence_threshold} "
            f"persona={persona_name!r}"
        )

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        uniform = 1.0 / len(self.hypothesis_space)
        return {title: uniform for title in self.hypothesis_space}

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts: list[str] = []

        parts.append(
            dedent(
                f"""\
                You are an insightful film curator collaborating with a movie lover to assemble a personalised watchlist consisting of 10 movies.

                ### Candidate Films
                {self._candidate_block}

                Use the user's answers to understand their taste. Ask concise YES/NO questions that reveal concrete preferences about tone, pacing, genre blends, themes, or iconic elements. Avoid repeating or paraphrasing earlier questions.
                """
            ).strip()
        )

        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(
                f"- Q: {question}; A: {answer}" for question, answer in history
            )
            parts.append(
                dedent(
                    f"""
                    These are the questions asked so far and the user's answers:
                    {history_formatted}
                    """
                ).strip()
            )

        belief_state_formatted = "\n".join(
            f"- Film: {title}; Probability: {prob:.3f}"
            for title, prob in sorted(
                current_node.belief_state.items(), key=lambda item: item[1], reverse=True
            )
        )

        if belief_state_formatted:
            parts.append(
                dedent(
                    f"""\
                    Current belief distribution over the films:
                    {belief_state_formatted}
                    """
                ).strip()
            )

        parts.append(
            dedent(
                f"""
                Your task is to generate {self.max_question_nodes} excellent YES/NO questions to ask the user next.
                Remember, the goal is to recommend 10 films from the candidate list. The questions should help you understand the user's preferences better, not narrow down to a single film.

                Format your response exactly as:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """
            ).strip()
        )

        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)
        questions = parse_binary_questions(output)

        return {question.question: question.possible_answers for question in questions}

    @override
    @retry(stop=stop_after_attempt(2))
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        prompt = dedent(
            f"""\
            You are estimating how likely a user would answer "Yes" to a question if each film below ended up being in your final 10 recommendations.

            ### Candidate Films
            {self._candidate_block}

            ### Question
            "{question}"

            ### Task
            - For each film, assume it is the correct recommendation for the user (part of the final 10 recommendations).
            - Estimate the probability (0.0 to 1.0) that the user would answer "Yes".
            - Probabilities must be rounded to two decimals.
            - Use only the exact film titles.

            ### Response Format
            1. <Film Title>|<probability>
            2. <Film Title>|<probability>
            ...

            Do not include commentary or extra text.
            """
        ).strip()

        output = await query_llm(prompt, self.questioner_session)
        probabilities = parse_probabilities(output)

        likelihood_map: dict[str, dict[str, float]] = {}
        for probability in probabilities:
            if probability.hypothesis not in hypotheses:
                continue
            yes_prob = max(0.0, min(1.0, probability.probability))
            likelihood_map[probability.hypothesis] = {
                "Yes": yes_prob,
                "No": 1.0 - yes_prob,
            }

        missing = set(hypotheses) - set(likelihood_map)
        if missing:
            for hypothesis in missing:
                likelihood_map[hypothesis] = {"Yes": 0.5, "No": 0.5}

        return likelihood_map

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        persona_name = self.instance.persona.name or "the movie lover"
        gender_note = (
            f" {self.instance.persona.gender.lower()}"
            if self.instance.persona.gender
            else ""
        )
        options_block = ", ".join(
            f'"{answer}"' for answer in current_node.possible_answers
        )

        prompt = dedent(
            f"""\
            You are {persona_name}{gender_note}. Stay fully in character and rely on the persona briefing when answering.

            Persona briefing:
            {self._persona_block}

            Available responses: {options_block}

            ### Question
            "{current_node.question}"

            Reply with EXACTLY one of the available responses—no additional words.
            """
        ).strip()

        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)
