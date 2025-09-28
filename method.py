import asyncio
from datetime import datetime
from functools import partial
import logging
from history import RunRecord, serialise_evidence_node
from node import EvidenceNode, QuestionNode, get_conversation_depth
from question_clustering import QuestionClustering
from rewards import expected_reward
from tasks.task import Task

logger = logging.getLogger("Method")


async def run_task(
    task: Task,
    question_clustering: QuestionClustering,
    sharpness_constant: float,
    min_probability: float,
) -> RunRecord:
    start_time = datetime.now()
    final_path: list[str] = []

    # Create root node
    initial_belief_state = await task.create_initial_belief_state()
    root = EvidenceNode("ROOT", initial_belief_state, 1.0)
    logger.info(f"Created root: {str(root)}")
    current_node = root
    final_path.append(str(current_node))

    while not is_terminal(current_node, task):
        await expand_evidence(
            current_node, 0, task, question_clustering, min_probability
        )

        best_question_node = max(
            current_node.children,
            key=partial(expected_reward, sharpness_constant=sharpness_constant),
        )
        logger.info(f"Selected question node: {str(best_question_node)}")
        final_path.append(str(best_question_node))

        # Get question answer and move ahead
        selected_evidence_node = await task.get_answer(best_question_node)
        logger.info(f"Benchmark LLM selected: {str(selected_evidence_node)}")
        current_node = selected_evidence_node
        final_path.append(str(current_node))

    end_time = datetime.now()
    logger.info(
        f"Completed run in {end_time - start_time}s! "
        f"Final belief: {current_node.belief_state}"
    )

    return RunRecord(
        task_info=str(task),
        questioner_session=task.questioner_session,
        answerer_session=task.answerer_session,
        expected_answer=task.task_answer,
        start_time=start_time,
        end_time=end_time,
        final_path=final_path,
        final_belief_state=current_node.belief_state,
        serialised_tree=serialise_evidence_node(root),
    )


async def expand_evidence(
    current_node: EvidenceNode,
    current_depth: int,
    task: Task,
    question_clustering: QuestionClustering,
    min_probability: float,
) -> None:
    if is_terminal(current_node, task) or current_depth >= task.max_lookahead_depth:
        return

    if not current_node.children:
        new_questions = await task.create_questions(current_node)
        new_question_nodes = [QuestionNode(q, current_node) for q in new_questions]
        current_node.children.extend(new_question_nodes)

    await asyncio.gather(
        *[
            expand_questions(
                child, current_depth, task, question_clustering, min_probability
            )
            for child in current_node.children
        ]
    )


async def expand_questions(
    current_node: QuestionNode,
    current_depth: int,
    task: Task,
    question_clustering: QuestionClustering,
    min_probability: float,
) -> None:
    if not current_node.children:
        cluster = question_clustering.get_cluster(current_node.question)

        async with cluster.lock:
            missing_hypotheses = set(current_node.parent.belief_state.keys()) - set(
                cluster.get_hypotheses()
            )

            if missing_hypotheses:
                new_likelihoods = await task.get_likelihoods(
                    current_node.question, list(missing_hypotheses)
                )
                cluster.likelihoods.update(new_likelihoods)

        for answer in cluster.get_answers():
            likelihoods = cluster.get_likelihoods_for_answer(answer)
            posterior, marginal = calculate_posterior(
                current_node.parent.belief_state, likelihoods, min_probability
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
                child, current_depth + 1, task, question_clustering, min_probability
            )
            for child in current_node.children
        ]
    )


def calculate_posterior(
    prior: dict[str, float], likelihoods: dict[str, float], min_probability: float
) -> tuple[dict[str, float], float]:
    all_posteriors = {h: p * likelihoods.get(h, 1.0) for h, p in prior.items()}

    # Warn for missing likelihoods
    for h in prior:
        if h not in likelihoods:
            logger.warning(f"{h} not found in likelihoods! Defaulting to 1...")

    unnormalised = {h: p for h, p in all_posteriors.items() if p >= min_probability}

    # Ensure at least one hypothesis. This is just for sanity so other functions
    # don't break. The marginal will be close to zero, so rewards shouldn't be
    # affected.
    if not unnormalised:
        max_h = max(all_posteriors, key=all_posteriors.__getitem__)
        unnormalised[max_h] = all_posteriors[max_h]

    marginal = sum(unnormalised.values())
    normalised = {h: p / marginal for h, p in unnormalised.items()}
    return normalised, marginal


def is_terminal(node: EvidenceNode, task: Task) -> bool:
    return get_conversation_depth(node) >= task.max_conversation_depth or any(
        prob >= task.confidence_threshold for prob in node.belief_state.values()
    )
