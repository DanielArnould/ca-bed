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

from .common import (
    PersonaLineup,
    QUESTIONER_ROLE_PROMPT,
    build_answerer_prompt,
    build_question_evaluation_header,
    build_question_generation_instructions,
    build_questioner_preamble,
    create_persona_lineup,
    format_belief_block,
    format_history_block,
    format_persona_list,
    format_target_movie,
)
from .data import PersonaMovieMatchInstance


class Bayesian(Task):
    lineup: PersonaLineup

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: PersonaMovieMatchInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
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
            confidence_threshold=confidence_threshold,
            hypothesis_space=self.lineup.hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            "MovieLens Persona Match (Bayesian): "
            f"answer={self.task_answer!r} "
            f"max_question_nodes={self.max_question_nodes} "
            f"max_lookahead_depth={self.max_lookahead_depth} "
            f"max_conversation_depth={self.max_conversation_depth} "
            f"confidence_threshold={self.confidence_threshold}"
        )

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        persona_briefings = format_persona_list(
            self.lineup,
            is_answerer=False,
            numbered=False,
        )

        prompt = dedent(
            f"""\
            {QUESTIONER_ROLE_PROMPT}

            Before any interrogation, draft a prior belief about which persona is the hidden fan of the target film.

            ### Target Film
            {format_target_movie(self.lineup.target_movie)}

            ### Persona Briefings
            {persona_briefings}

            ### Task
            - Evaluate how strongly each persona is likely to love the target film based on their tastes, motifs, dislikes, and reference favourites.
            - Assign a probability to every persona listed above; none may be omitted.
            - Probabilities must sum to 1.0 (±0.01 tolerance).
            - Use the persona names exactly as displayed in the briefings.
            - Express each probability as a decimal rounded to two places.
            - Return only the formatted response; no commentary.

            ### Response Format
            1. <Persona Name>|<probability>
            2. <Persona Name>|<probability>
            """
        ).strip()

        output = await query_llm(prompt, self.questioner_session)
        priors = parse_probabilities(output)
        belief_state = {prior.hypothesis: prior.probability for prior in priors}
        missing = [name for name in self.hypothesis_space if name not in belief_state]

        if missing:
            raise RuntimeError(f'Missing likelihoods for: {missing}')

        return belief_state

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

        positive_label = answers[0] if answers else "Yes"
        negative_label = (
            answers[1] if len(answers) > 1 else ("No" if answers else "No")
        )

        instruction = dedent(
            f"""\
            ### Task
            - Assume each persona listed is the hidden fan of the target film.
            - Estimate the probability (0.0 to 1.0) that they would answer "{positive_label}" to the question.
            - Use the persona id exactly as provided.
            - Return only the formatted response; no explanations or commentary.

            ### Response Format
            One line per persona:
            <number>. <EXACT Persona-id>|<probability>
            """
        ).strip()

        prompt = "\n\n".join([header, instruction])
        output = await query_llm(prompt, self.questioner_session)
        probabilities = parse_probabilities(output)

        likelihood_map: dict[str, dict[str, float]] = {}
        for probability in probabilities:
            yes_prob = max(0.0, min(1.0, probability.probability))
            likelihood_map[probability.hypothesis] = {
                positive_label: yes_prob,
                negative_label: 1.0 - yes_prob,
            }

        missing = set(hypotheses) - set(likelihood_map)
        if missing:
            raise RuntimeError(f'Missing likelihoods for: {missing}')

        return likelihood_map

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        prompt = build_answerer_prompt(self.lineup, current_node.question)
        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)
