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
    parse_categorical_likelihoods,
)

from .common import MovieLensInstance, format_candidate_movies, format_persona_context


class Baseline(Task):
    instance: MovieLensInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: MovieLensInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
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
            confidence_threshold=1.0,
            hypothesis_space=instance.hypothesis_space,
        )

    def __str__(self) -> str:
        persona_name = self.instance.persona.name or "Unknown Persona"
        return (
            "MovieLens (Non-Bayesian): "
            f"task_answer={self.task_answer!r} "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} "
            f"max_conversation_depth={self.max_conversation_depth} "
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
                You are an expert film curator in a live conversation with a movie lover. Your goal is to discover which film from the candidate list below to recommend based solely on the dialogue.

                ### Candidate Films
                {self._candidate_block}
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
                    These are the questions asked so far and the user's replies:
                    {history_formatted}
                    """
                ).strip()
            )

        belief_state_formatted = "\n".join(
            f"- {title}"
            for title, _ in sorted(
                current_node.belief_state.items(), key=lambda item: item[1], reverse=True
            )
        )

        if belief_state_formatted:
            parts.append(
                dedent(
                    f"""\
                    Based on your current beliefs, the leading candidates are:
                    {belief_state_formatted}
                    """
                ).strip()
            )

        parts.append(
            dedent(
                f"""
                Your task is to generate {self.max_question_nodes} excellent YES/NO questions to ask the user next.
                The best questions help you learn their tastes from scratch and sharply distinguish between the remaining candidate films using concrete cinematic traits, plot points, themes, or tonal qualities.

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
        answers_block = "\n".join(f"- {answer}" for answer in answers)

        prompt = dedent(
            f"""\
            You are evaluating which answer each candidate film would produce to the following YES/NO question, assuming that film is the one ultimately recommended.

            ### Candidate Films
            {self._candidate_block}

            ### Question
            "{question}"

            ### Possible Answers
            {answers_block}

            ### Task
            - Assume each film listed is the true recommendation for the user.
            - Decide which answer (exactly one) that film would yield.
            - Cover every film exactly once.
            - Use ONLY the exact film titles as given.

            ### Response Format
            Yes: Title A, Title B, ...
            Count of 'Yes': <integer>
            No: Title C, Title D, ...
            Count of 'No': <integer>

            Do not include commentary or extra text.
            """
        ).strip()

        output = await query_llm(prompt, self.questioner_session)
        likelihoods = parse_categorical_likelihoods(output, possible_answers=answers)

        likelihood_map: dict[str, dict[str, float]] = {}
        for likelihood in likelihoods:
            if likelihood.hypothesis not in hypotheses:
                continue
            likelihood_map[likelihood.hypothesis] = {
                answer: probability
                for answer, probability in zip(answers, likelihood.likelihoods, strict=True)
            }

        missing = set(hypotheses) - set(likelihood_map)
        if missing:
            default = {answer: 1.0 / len(answers) for answer in answers}
            for hypothesis in missing:
                likelihood_map[hypothesis] = default.copy()

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
