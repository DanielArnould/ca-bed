import asyncio
from datetime import datetime
import logging
from history import RunRecord, serialise_tree
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from question_clustering import QuestionClustering
from rewards import expected_reward
from tasks.task import Task

LOGGER = logging.getLogger(__name__)


class Method:
    benchmark_model: Model
    method_model: Model
    question_clustering: QuestionClustering
    root: EvidenceNode

    def __init__(
        self,
        benchmark_model: Model,
        method_model: Model,
        question_clustering: QuestionClustering,
        initial_belief_state: dict[str, float],
    ):
        self.benchmark_model = benchmark_model
        self.method_model = method_model
        self.question_clustering = question_clustering
        self.root = EvidenceNode(
            answer="ROOT",
            belief_state=initial_belief_state,
            marginal_likelihood=1.0,
        )
        LOGGER.info(f"Created root node: {str(self.root)}")

    async def run(self, task: Task) -> RunRecord:
        start_time = datetime.now()
        final_path = []

        current_node = self.root
        final_path.append(str(current_node))

        while not self.is_terminal(current_node, task):
            await self.expand_evidence(current_node, task, 0)
            best_question_node = max(current_node.children, key=expected_reward)
            LOGGER.info(f"Selected question node: {str(best_question_node)}")
            final_path.append(str(best_question_node))

            selection_prompt = task.get_answer_selection_prompt(best_question_node)
            LOGGER.info("Posing question to Benchmark LLM...")
            selection_output = await call_llm(selection_prompt, self.benchmark_model)
            selected_evidence_node = task.parse_answer_selection_output(
                selection_output, best_question_node
            )
            LOGGER.info(f"Benchmark LLM selected: {str(selected_evidence_node)}")
            current_node = selected_evidence_node

            final_path.append(str(current_node))

        best_guess = max(
            current_node.belief_state,
            key=current_node.belief_state.__getitem__,
        )

        LOGGER.info(
            f"Completed run! Best guess: {best_guess}, Target: {task.task_answer}"
        )

        return RunRecord(
            task_info=str(task),
            true_answer=task.task_answer,
            start_time=start_time,
            end_time=datetime.now(),
            serialised_tree=serialise_tree(self.root),
            final_path=final_path,
            final_answer=best_guess,
        )

    async def expand_evidence(
        self, curr: EvidenceNode, task: Task, current_depth: int
    ) -> None:
        if self.is_terminal(curr, task) or current_depth >= task.max_lookahead_depth:
            return

        if not curr.children:
            prompt = task.get_question_generation_prompt(curr)
            output = await call_llm(prompt, self.method_model)
            new_questions = task.parse_question_generation_output(output)
            new_question_nodes = [QuestionNode(q, curr) for q in new_questions]
            curr.children.extend(new_question_nodes)

        await asyncio.gather(
            *[
                self.expand_questions(child, task, current_depth)
                for child in curr.children
            ]
        )

    async def expand_questions(
        self, curr: QuestionNode, task: Task, current_depth: int
    ) -> None:
        if not curr.children:
            cluster = self.question_clustering.get_cluster(curr.question)

            async with cluster.lock:
                if cluster.likelihoods is None:
                    output = await call_llm(
                        task.get_likelihood_elicitation_prompt(curr.question),
                        self.benchmark_model,
                    )
                    cluster.likelihoods = task.parse_likelihood_elicitation_output(
                        output
                    )

            for answer, likelihoods in cluster.likelihoods.items():
                posterior, marginal = self.calculate_posterior(
                    curr.parent.belief_state, likelihoods
                )
                evidence_node = EvidenceNode(
                    answer=answer,
                    belief_state=posterior,
                    marginal_likelihood=marginal,
                    parent=curr,
                )
                curr.children.append(evidence_node)

        await asyncio.gather(
            *[
                self.expand_evidence(child, task, current_depth + 1)
                for child in curr.children
            ]
        )

    def calculate_posterior(
        self, prior: dict[str, float], likelihoods: dict[str, float]
    ) -> tuple[dict[str, float], float]:
        # Calculate unnormalised posterior: P(hypothesis) * P(evidence|hypothesis)
        unnormalised_posterior: dict[str, float] = {}
        for hypo, prior_belief in prior.items():
            likelihood = likelihoods.get(hypo, 1.0)
            if hypo not in likelihoods:
                LOGGER.warning(f"{hypo} not found in likelihoods! Defaulting to 1...")
            unnormalised_posterior[hypo] = prior_belief * likelihood

        # Calculate marginal (normalisation constant)
        marginal = sum(unnormalised_posterior.values())

        # We can only get a 0 marginal if the parent has a zeroed belief state (impossible)
        # or the likelihoods are all 0, in which case it's justified to make a zeroed belief state
        if marginal == 0:
            LOGGER.warning(
                "Marginal likelihood of 0, creating a zeroed belief state..."
            )
            posterior = {hypo: 0.0 for hypo in unnormalised_posterior}
        else:
            posterior = {
                hypo: (prob / marginal) for hypo, prob in unnormalised_posterior.items()
            }

        return posterior, marginal

    def is_terminal(self, node: EvidenceNode, task: Task) -> bool:
        return get_conversation_depth(node) >= task.max_conversation_depth or any(
            prob >= task.confidence_threshold for prob in node.belief_state.values()
        )


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


def get_uniform_belief_state(hypothesis_space: list[str]) -> dict[str, float]:
    prob = 1.0 / len(hypothesis_space)
    return {item: prob for item in hypothesis_space}
