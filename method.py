import asyncio
import copy
from datetime import datetime
from itertools import chain
import logging
from history import RunHistory
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from question_clustering import Cluster, QuestionClustering
from rewards import expected_reward
from tasks.task import Task

LOGGER = logging.getLogger("Method")


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

    async def run(self, task: Task) -> RunHistory:
        start_time = datetime.now()
        final_path = []

        current_node = self.root
        final_path.append(str(current_node))

        while not self.is_terminal(current_node, task):
            await self.expand_evidence([current_node], task, 0)
            best_question_node = max(current_node.children, key=expected_reward)
            LOGGER.info(f"Selected question node: {str(best_question_node)}")
            final_path.append(str(best_question_node))

            selection_prompt = task.get_answer_selection_prompt(best_question_node)
            LOGGER.info("Posing question to Benchmark LLM...")
            selection_output = await call_llm(selection_prompt, self.benchmark_model)
            selected_evidence_node = task.parse_answer_selection_output(
                selection_output.string, best_question_node
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

        # UNSAFE RETURN, since tree and question clustering could be mutated
        # Voyager has issues being deepcopied though
        return RunHistory(
            task_info=str(task),
            actual_answer=task.task_answer,
            start_time=start_time,
            end_time=datetime.now(),
            tree=self.root,
            final_path=final_path,
            final_answer=best_guess,
            question_clustering=self.question_clustering,
        )

    async def expand_evidence(
        self, level: list[EvidenceNode], task: Task, current_depth: int
    ) -> None:
        # Filter out terminal nodes and check if reached lookahead limit
        active = [node for node in level if not self.is_terminal(node, task)]
        if current_depth >= task.max_lookahead_depth or not active:
            return

        # Separate nodes that need questions from those that already have children
        needs_questions = [node for node in active if not node.children]
        next_level: list[QuestionNode] = list(
            chain.from_iterable(node.children for node in active if node.children)
        )

        # Generate questions only for nodes without children
        if needs_questions:
            prompts = [
                task.get_question_generation_prompt(node) for node in needs_questions
            ]
            outputs = await asyncio.gather(
                *[call_llm(p, self.method_model) for p in prompts]
            )
            new_questions = [
                task.parse_question_generation_output(out.string) for out in outputs
            ]

            # Create question nodes for nodes that needed them
            for questions, parent in zip(new_questions, needs_questions):
                new_question_nodes = [QuestionNode(q, parent) for q in questions]
                parent.children.extend(new_question_nodes)
                next_level.extend(new_question_nodes)

        await self.expand_questions(next_level, task, current_depth)

    async def expand_questions(
        self, level: list[QuestionNode], task: Task, current_depth: int
    ) -> None:
        # Separate nodes that need evidence from those that already have children
        needs_evidence = [node for node in level if not node.children]
        next_level: list[EvidenceNode] = list(
            chain.from_iterable(node.children for node in level if node.children)
        )

        # Only process nodes that need evidence generation
        if needs_evidence:
            # Get or create question clusters for nodes that need evidence.
            # New clusters are created sequentially, so items in the same list
            # may join clusters created by previous items
            clusters = [
                self.question_clustering.get_cluster(node.question)
                for node in needs_evidence
            ]
            seen_ids = set()
            new_clusters: list[Cluster] = []
            for cluster in clusters:
                if cluster.likelihoods is None and id(cluster) not in seen_ids:
                    seen_ids.add(id(cluster))
                    new_clusters.append(cluster)

            # Calculate likelihoods for new clusters concurrently
            if new_clusters:
                outputs = await asyncio.gather(
                    *[
                        call_llm(
                            task.get_likelihood_elicitation_prompt(c.centroid_question),
                            self.method_model,
                        )
                        for c in new_clusters
                    ]
                )

                for output, cluster in zip(outputs, new_clusters):
                    cluster.likelihoods = task.parse_likelihood_elicitation_output(
                        output.string
                    )

            # Create evidence nodes for question nodes that needed them
            for question_node, cluster in zip(needs_evidence, clusters):
                assert cluster.likelihoods is not None, "Likelihoods missing!"

                for answer, likelihoods in cluster.likelihoods.items():
                    posterior, marginal = self.calculate_posterior(
                        question_node.parent.belief_state, likelihoods
                    )
                    evidence_node = EvidenceNode(
                        answer=answer,
                        belief_state=posterior,
                        marginal_likelihood=marginal,
                        parent=question_node,
                    )
                    question_node.children.append(evidence_node)
                    next_level.append(evidence_node)

        await self.expand_evidence(next_level, task, current_depth + 1)

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
