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

from .common import (
    PersonaLineup,
    build_answerer_prompt,
    build_question_evaluation_header,
    build_question_generation_instructions,
    build_questioner_preamble,
    create_persona_lineup,
    format_belief_block,
    format_history_block,
)
from .data import PersonaMovieMatchInstance


class Baseline(Task):
    lineup: PersonaLineup

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: PersonaMovieMatchInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
    ) -> None:
        self.lineup = create_persona_lineup(instance)
        self._questioner_preamble = build_questioner_preamble(self.lineup)

        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=self.lineup.target_name,
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=1.0,
            hypothesis_space=self.lineup.hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            "MovieLens Persona Match (Baseline): "
            f"answer={self.task_answer!r} "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} "
            f"max_conversation_depth={self.max_conversation_depth}"
        )

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        uniform = 1.0 / len(self.hypothesis_space)
        return {name: uniform for name in self.hypothesis_space}

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts: list[str] = [self._questioner_preamble]

        history_block = format_history_block(get_conversation_history(current_node))
        if history_block:
            parts.append(history_block)

        belief_block = format_belief_block(current_node.belief_state)
        if belief_block:
            parts.append(belief_block)

        parts.append(
            build_question_generation_instructions(self.max_question_nodes)
        )

        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)
        questions = parse_binary_questions(output)

        return {question.question: question.possible_answers for question in questions}

    @override
    @retry(stop=stop_after_attempt(2))
    async def get_likelihoods(
        self,
        question: str,
        answers: list[str],
        hypotheses: list[str],
    ) -> dict[str, dict[str, float]]:
        header = build_question_evaluation_header(
            self.lineup, question, answers, hypotheses
        )
        instruction = dedent(
            """\
            ### Task
            - Assume each persona listed is the hidden fan of the target film.
            - Decide which answer they would most likely give to the question.
            - Assign every persona to exactly one answer option (no omissions, no duplicates).
            - Use the persona names exactly as provided.
            - Present the answers in the same order that the answer options were listed.

            ### Response Format
            <Answer Label>: Persona 1, Persona 2, ...
            Count of '<Answer Label>': <integer>
            <Next Answer Label>: Persona 3, Persona 4, ...
            Count of '<Next Answer Label>': <integer>

            Do not include commentary or extra text."""
        ).strip()

        prompt = "\n\n".join([header, instruction])
        output = await query_llm(prompt, self.questioner_session)
        likelihoods = parse_categorical_likelihoods(output, possible_answers=answers)

        likelihood_map: dict[str, dict[str, float]] = {}
        for likelihood in likelihoods:
            likelihood_map[likelihood.hypothesis] = {
                answer: probability
                for answer, probability in zip(
                    answers, likelihood.likelihoods, strict=True
                )
            }

        missing = set(hypotheses) - set(likelihood_map)
        if missing:
            default = {answer: 1.0 / len(answers) for answer in answers}
            for hypothesis in missing:
                likelihood_map[hypothesis] = default.copy()

        return likelihood_map

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        prompt = build_answerer_prompt(self.lineup, current_node.question)
        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)
