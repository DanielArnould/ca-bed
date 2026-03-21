import asyncio
import math

from loguru import logger

from ca_bed.history import RunRecord
from ca_bed.node import (
    EvidenceNode,
    Likelihoods,
    ProbabilityDistribution,
    QuestionNode,
    get_conversation_depth,
    get_conversation_history,
)
from ca_bed.tasks.task import TreeBasedTask


async def run_tree_based_task(
    task: TreeBasedTask,
) -> RunRecord:
    initial_belief_state = task.create_uniform_belief_state()
    root = EvidenceNode(
        answer="ROOT", belief_state=initial_belief_state, marginal_likelihood=1.0
    )
    final_path = []
    current_node: EvidenceNode = root

    try:
        while not is_terminal(
            current_node, task.max_conversation_depth, task.confidence_threshold
        ):
            final_path.append(current_node)
            await expand_evidence(
                current_node=current_node,
                current_depth=0,
                task=task,
            )

            best_question_node = max(
                current_node.children, key=expected_information_gain
            )
            answer = await task.get_answer(
                best_question_node.question,
                best_question_node.possible_answers,
            )
            current_node = next(
                child for child in best_question_node.children if child.answer == answer
            )
    except Exception:
        logger.exception("A fatal error in the method occurred")

    final_path.append(current_node)
    return RunRecord(task=task, final_path=final_path)


async def expand_evidence(
    current_node: EvidenceNode,
    current_depth: int,
    task: TreeBasedTask,
) -> None:
    if (
        is_terminal(
            current_node, task.max_conversation_depth, task.confidence_threshold
        )
        or current_depth >= task.max_lookahead_depth
    ):
        return

    if not current_node.children:
        new_questions = await task.create_questions(
            get_conversation_history(current_node), current_node.belief_state
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
            )
            for child in current_node.children
        ]
    )


async def expand_questions(
    current_node: QuestionNode,
    current_depth: int,
    task: TreeBasedTask,
) -> None:
    if not current_node.children:
        new_likelihoods = await task.get_likelihoods(
            current_node.question, current_node.possible_answers
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
            )
            for child in current_node.children
        ]
    )


def calculate_posterior(
    prior: ProbabilityDistribution,
    likelihoods: Likelihoods,
) -> tuple[ProbabilityDistribution, float]:
    for h in prior:
        if h not in likelihoods:
            logger.warning(f"{h} not found in likelihoods!")
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
        or sum(node.belief_state.values()) == 0
    )


def shannon_entropy(belief_state: ProbabilityDistribution) -> float:
    return -sum(prob * math.log2(prob) for prob in belief_state.values() if prob > 0)


def expected_final_entropy(node: EvidenceNode | QuestionNode) -> float:
    """
    Recursively calculates the expected entropy at the leaf nodes of a given path.
    """
    match node:
        case QuestionNode():
            # Expected entropy is the weighted sum of the entropies of possible answers
            return sum(
                evidence.marginal_likelihood * expected_final_entropy(evidence)
                for evidence in node.children
            )

        case EvidenceNode():
            if node.children:
                # If we can ask more questions, pick the question that minimizes
                # the expected final entropy.
                return min(expected_final_entropy(child) for child in node.children)

            # Base case (leaf node): The actual entropy at this terminating state
            return shannon_entropy(node.belief_state)


def expected_information_gain(candidate_question: QuestionNode) -> float:
    """
    Calculates the Expected Information Gain of asking a question,
    accounting for the full lookahead depth.
    """
    parent = candidate_question.parent
    starting_entropy = shannon_entropy(parent.belief_state)

    # Information Gain = H(start) - H(expected_final)
    return starting_entropy - expected_final_entropy(candidate_question)
