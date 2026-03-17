import asyncio

from loguru import logger

from ca_bed.history import RunRecord
from ca_bed.llm import LLM
from ca_bed.node import (
    EvidenceNode,
    Likelihoods,
    ProbabilityDistribution,
    QuestionNode,
    get_conversation_depth,
    get_conversation_history,
)
from ca_bed.rewards import reward
from ca_bed.tasks.task import Task


async def run_task(
    task: Task,
    questioner_llm: LLM,
    answerer_llm: LLM,
    n_questions: int,
    max_conversation_depth: int,
    max_lookahead_depth: int,
    confidence_threshold: float,
) -> RunRecord:
    initial_belief_state = await task.create_initial_belief_state()
    root = EvidenceNode(
        answer="ROOT", belief_state=initial_belief_state, marginal_likelihood=1.0
    )
    final_path = []
    current_node: EvidenceNode = root

    try:
        while not is_terminal(
            current_node, max_conversation_depth, confidence_threshold
        ):
            final_path.append(current_node)
            await expand_evidence(
                current_node=current_node,
                current_depth=0,
                task=task,
                questioner_llm=questioner_llm,
                n_questions=n_questions,
                max_conversation_depth=max_conversation_depth,
                confidence_threshold=confidence_threshold,
                max_lookahead_depth=max_lookahead_depth,
            )

            best_question_node = max(current_node.children, key=reward)
            answer = await task.get_answer(
                best_question_node.question,
                best_question_node.possible_answers,
                answerer_llm,
            )
            current_node = next(
                child for child in best_question_node.children if child.answer == answer
            )
    except Exception:
        logger.exception("What!?")

    return RunRecord(task=task, final_path=final_path)


async def expand_evidence(
    current_node: EvidenceNode,
    current_depth: int,
    task: Task,
    questioner_llm: LLM,
    n_questions: int,
    max_conversation_depth: int,
    confidence_threshold: float,
    max_lookahead_depth: int,
) -> None:
    if (
        is_terminal(current_node, max_conversation_depth, confidence_threshold)
        or current_depth >= max_lookahead_depth
    ):
        return

    if not current_node.children:
        new_questions = await task.create_questions(
            get_conversation_history(current_node), n_questions, questioner_llm
        )
        new_question_nodes = [
            QuestionNode(q, answers, current_node)
            for q, answers in new_questions.items()
        ]
        current_node.children.extend(new_question_nodes)

    await asyncio.gather(
        *[
            expand_questions(
                child,
                current_depth,
                task,
                questioner_llm,
                n_questions,
                max_conversation_depth,
                confidence_threshold,
                max_lookahead_depth,
            )
            for child in current_node.children
        ]
    )


async def expand_questions(
    current_node: QuestionNode,
    current_depth: int,
    task: Task,
    questioner_llm: LLM,
    n_questions: int,
    max_conversation_depth: int,
    confidence_threshold: float,
    max_lookahead_depth: int,
) -> None:
    if not current_node.children:
        new_likelihoods = await task.get_likelihoods(
            current_node.question, current_node.possible_answers, questioner_llm
        )

        for answer, likelihoods in new_likelihoods.items():
            posterior, marginal = calculate_posterior(
                current_node.parent.belief_state,
                likelihoods,
            )
            evidence_node = EvidenceNode(
                answer=answer,
                belief_state=posterior,
                marginal_likelihood=marginal,
                parent=current_node,
            )
            current_node.children.append(evidence_node)

    await asyncio.gather(
        *[
            expand_evidence(
                child,
                current_depth + 1,
                task,
                questioner_llm,
                n_questions,
                max_conversation_depth,
                confidence_threshold,
                max_lookahead_depth,
            )
            for child in current_node.children
        ]
    )


def calculate_posterior(
    prior: ProbabilityDistribution,
    likelihoods: Likelihoods,
) -> tuple[ProbabilityDistribution, float]:
    unnormalized = {h: p * likelihoods.get(h, 0.0) for h, p in prior.items()}
    marginal = sum(unnormalized.values())
    normalized_posteriors = {}
    if marginal > 0:
        normalized_posteriors = {h: p / marginal for h, p in unnormalized.items()}

    return normalized_posteriors, marginal


def is_terminal(
    node: EvidenceNode, max_conversation_depth: int, confidence_threshold: float
) -> bool:
    return (
        get_conversation_depth(node) >= max_conversation_depth
        or any(prob >= confidence_threshold for prob in node.belief_state.values())
        or len(node.belief_state) == 0
    )
