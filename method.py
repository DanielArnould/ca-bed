import asyncio
import copy
from datetime import datetime
from logging import Logger
from experiment_logging import RunHistory
from loggers import get_logger
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from question_clustering import QuestionClustering
from rewards import expected_reward
from tasks.task import Question, Task


class Method:
    # Constants
    model: Model
    question_clustering: QuestionClustering
    root: EvidenceNode

    def __init__(
        self,
        model: Model,
        question_clustering: QuestionClustering,
        initial_belief_state: dict[str, float],
    ):
        self.model = model
        self.question_clustering = question_clustering
        self.root = EvidenceNode(
            answer="ROOT",
            belief_state=initial_belief_state,
            marginal_likelihood=1.0,
        )

    async def run(self, task: Task) -> None:
        current_node = self.root

        while not self.is_terminal(current_node, task):
            await self.lookahead(current_node, 0, task)
            best_question_node = max(current_node.children, key=expected_reward)

            selection_prompt = task.get_answer_selection_prompt(best_question_node)
            selection_output = await call_llm(selection_prompt, self.model)
            selected_evidence_node = task.parse_answer_selection_output(
                selection_output.string, best_question_node
            )
            current_node = selected_evidence_node

        best_guess = max(
            current_node.belief_state,
            key=current_node.belief_state.__getitem__,
        )

        return None

    async def lookahead(self, node: EvidenceNode, curr_depth: int, task: Task) -> None:
        # Should we continue any further?
        if curr_depth >= task.max_lookahead_depth or self.is_terminal(node, task):
            return

        # Generate questions
        if len(node.children) == 0:
            question_gen_prompt = task.get_question_generation_prompt(node)
            question_gen_output = await call_llm(question_gen_prompt, self.model)
            questions = task.parse_question_generation_output(
                question_gen_output.string
            )

            create_question_node_tasks = [
                asyncio.create_task(self.create_question_node(question, node, task))
                for question in questions
            ]

            processed_question_nodes = await asyncio.gather(*create_question_node_tasks)
            node.children.extend(processed_question_nodes)

        recursive_tasks = [
            self.lookahead(child, curr_depth + 1, task)
            for question in node.children
            for child in question.children
        ]
        await asyncio.gather(*recursive_tasks)

    async def create_question_node(
        self, question: Question, parent_node: EvidenceNode, task: Task
    ) -> QuestionNode:
        question_node = QuestionNode(question=question.question, parent=parent_node)

        question_embedding = self.question_clustering.get_embedding(question.question)
        nearest_cluster = self.question_clustering.get_nearest_cluster(
            question_embedding
        )
        if nearest_cluster is None:
            likelihood_output = await call_llm(
                task.get_likelihood_elicitation_prompt(parent_node, question),
                self.model,
            )
            likelihoods_for_all_answers = task.parse_likelihood_elicitation_output(
                likelihood_output.string, question
            )
            self.question_clustering.add_cluster(
                question.question, question_embedding, likelihoods_for_all_answers
            )
        else:
            likelihoods_for_all_answers = nearest_cluster.likelihoods
            nearest_cluster.add_question(question.question, question_embedding)

        for answer, likelihoods_for_answer in likelihoods_for_all_answers.items():
            unnormalised_posterior: dict[str, float] = {}
            for hypo, belief in parent_node.belief_state.items():
                unnormalised_posterior[hypo] = belief * likelihoods_for_answer.get(
                    hypo, 0
                )

            marginal_likelihood = sum(unnormalised_posterior.values())
            posterior = {
                hypo: (prob / marginal_likelihood) if marginal_likelihood > 0 else 0
                for hypo, prob in unnormalised_posterior.items()
            }
            evidence_node = EvidenceNode(
                answer=answer,
                belief_state=posterior,
                marginal_likelihood=marginal_likelihood,
                parent=question_node,
            )
            question_node.children.append(evidence_node)

        return question_node

    def is_terminal(self, node: EvidenceNode, task: Task) -> bool:
        return get_conversation_depth(node) >= task.max_conversation_depth or any(
            prob >= task.confidence_threshold for prob in node.belief_state.values()
        )


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


if __name__ == "__main__":
    ...
