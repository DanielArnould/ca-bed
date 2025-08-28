"""
This module orchestrates the different methods of conversation employed
by enhanced UoT. It should handle the expansion and creation of the tree,
getting and processing the examiner response and the guesser response

**IMPORTANT**: As much info as possible should be saved with every run. This includes:
- Conversation history
- Tree state.
"""

import asyncio
import copy
from datetime import datetime
import logging
from experiment_logging import RunHistory
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from rewards import expected_reward
from tasks.task import Question, Task

LOGGER = logging.getLogger("Method")


class Method:
    model: Model
    task: Task
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float

    _current_node: EvidenceNode
    _root: EvidenceNode

    def __init__(
        self,
        model: Model,
        task: Task,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
    ):
        self.model = model
        self.task = task
        self.max_lookahead_depth = max_lookahead_depth
        self.max_conversation_depth = max_conversation_depth
        self.confidence_threshold = confidence_threshold

    async def run(self) -> RunHistory:
        start_time = datetime.now()
        tree_states = []
        final_path = []

        initial_belief_state = self.task.get_initial_belief_state()
        self._root = self._current_node = EvidenceNode(
            answer="ROOT", belief_state=initial_belief_state, marginal_likelihood=1.0
        )
        LOGGER.info(f"Created root node: {str(self._root)}")

        while not self._is_terminal(self._current_node):
            tree_states.append(copy.deepcopy(self._root))
            final_path.append(str(self._current_node))

            await self._lookahead(self._current_node, 0)
            best_question_node = max(self._current_node.children, key=expected_reward)
            LOGGER.info(f"Selected question node: {str(best_question_node)}")
            final_path.append(str(best_question_node))

            selection_prompt = self.task.get_answer_selection_prompt(best_question_node)
            LOGGER.info("Posing question to LLM...")
            selection_output = await call_llm(selection_prompt, self.model)
            selected_evidence_node = self.task.parse_answer_selection_output(
                selection_output.string, best_question_node
            )
            LOGGER.info(f"LLM selected: {str(selected_evidence_node)}")
            self._current_node = selected_evidence_node

        tree_states.append(copy.deepcopy(self._root))
        final_path.append(str(self._current_node))

        best_guess = max(
            self._current_node.belief_state,
            key=self._current_node.belief_state.__getitem__,
        )

        LOGGER.info(f"Completed run! Best guess: {best_guess}")

        return RunHistory(
            task_info=str(self.task),
            actual_answer=self.task.task_answer,
            start_time=start_time,
            end_time=datetime.now(),
            tree_states=tree_states,
            final_path=final_path,
            final_answer=best_guess,
        )

    async def _lookahead(self, node: EvidenceNode, curr_depth: int) -> None:
        # Should we continue any further?
        if curr_depth >= self.max_lookahead_depth or self._is_terminal(node):
            LOGGER.info(f"Ending lookahead at {str(node)}")
            return

        # Generate questions
        if len(node.children) == 0:
            LOGGER.info("Generating questions...")
            question_gen_prompt = self.task.get_question_generation_prompt(node)
            question_gen_output = await call_llm(question_gen_prompt, self.model)
            questions = self.task.parse_question_generation_output(
                question_gen_output.string
            )

            create_question_node_tasks = [
                asyncio.create_task(
                    self._create_question_node(
                        self.task.get_likelihood_elicitation_prompt(node, question),
                        question,
                        node,
                    )
                )
                for question in questions
            ]

            LOGGER.info(
                f"Creating {len(create_question_node_tasks)} question nodes concurrently..."
            )
            processed_question_nodes = await asyncio.gather(*create_question_node_tasks)
            node.children.extend(processed_question_nodes)

        recursive_tasks = [
            self._lookahead(child, curr_depth + 1)
            for question in node.children
            for child in question.children
        ]
        LOGGER.info(
            f"Performing recursive lookahead for {len(recursive_tasks)} new nodes concurrently..."
        )
        await asyncio.gather(*recursive_tasks)

    async def _create_question_node(
        self, prompt: str, question: Question, parent_node: EvidenceNode
    ) -> QuestionNode:
        question_node = QuestionNode(question=question.question, parent=parent_node)

        likelihood_output = await call_llm(prompt, self.model)
        likelihoods_for_all_answers = self.task.parse_likelihood_elicitation_output(
            likelihood_output.string, question
        )
        LOGGER.debug(
            f"Likelihoods for question '{question}' are {likelihoods_for_all_answers}"
        )

        for answer, likelihoods_for_answer in likelihoods_for_all_answers.items():
            unnormalised_posterior = {
                hypo: parent_node.belief_state[hypo] * likelihoods_for_answer[hypo]
                for hypo in parent_node.belief_state
            }
            marginal_likelihood = sum(unnormalised_posterior.values())
            posterior = {
                hypo: prob / marginal_likelihood
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

    def _get_conversation_depth(self, node: EvidenceNode) -> int:
        if node.parent is None:
            return 0

        return 1 + self._get_conversation_depth(node.parent.parent)

    def _is_terminal(self, node: EvidenceNode) -> bool:
        return self._get_conversation_depth(node) >= self.max_conversation_depth or any(
            prob >= self.confidence_threshold for prob in node.belief_state.values()
        )


if __name__ == "__main__":
    ...
