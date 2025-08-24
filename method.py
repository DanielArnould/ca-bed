"""
This module orchestrates the different methods of conversation employed
by enhanced UoT. It should handle the expansion and creation of the tree,
getting and processing the examiner response and the guesser response

**IMPORTANT**: As much info as possible should be saved with every run. This includes:
- Conversation history
- Tree state.
"""

from itertools import chain
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from rewards import expected_future_reward
from tasks.task import Task, InteractionMode


class Method:
    model: Model
    task: Task
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float

    _current_node: EvidenceNode

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

    def run(self) -> str:
        initial_belief_state = self.task.get_initial_belief_state()
        self._current_node = EvidenceNode(
            answer="ROOT", belief_state=initial_belief_state, marginal_likelihood=1.0
        )

        while not self._is_terminal(self._current_node):
            self._lookahead(self._current_node, 0)
            best_question_node = max(
                self._current_node.children, key=expected_future_reward
            )

            match self.task.interaction_mode:
                case InteractionMode.INTERACTIVE:
                    selected_evidence_node = self._pose_question(best_question_node)
                case InteractionMode.BENCHMARK:
                    selection_prompt = self.task.get_answer_selection_prompt(
                        best_question_node
                    )
                    selection_output = call_llm(selection_prompt, self.model)
                    selected_evidence_node = self.task.parse_answer_selection_output(
                        selection_output.string, best_question_node
                    )

            self._current_node = selected_evidence_node

        best_guess = max(
            self._current_node.belief_state,
            key=self._current_node.belief_state.__getitem__,
        )
        return best_guess

    def _lookahead(self, node: EvidenceNode, curr_depth: int) -> None:
        # Should we continue any further?
        if curr_depth >= self.max_lookahead_depth or self._is_terminal(node):
            return

        # Have we already looked ahead at this stage?
        if len(node.children) > 0:
            for child in chain.from_iterable(
                question.children for question in node.children
            ):
                self._lookahead(child, curr_depth + 1)

            return

        # Generate questions
        question_gen_prompt = self.task.get_question_generation_prompt(node)
        question_gen_output = call_llm(question_gen_prompt, self.model)
        questions = self.task.parse_question_generation_output(
            question_gen_output.string
        )

        for question in questions:
            question_node = QuestionNode(question=question.question, parent=node)
            likelihood_prompt = self.task.get_likelihood_elicitation_prompt(
                node, question
            )
            likelihood_output = call_llm(likelihood_prompt, self.model)
            likelihoods_for_all_answers = self.task.parse_likelihood_elicitation_output(
                likelihood_output.string, question
            )

            for answer, likelihoods_for_answer in likelihoods_for_all_answers.items():
                unnormalised_posterior = {
                    hypo: node.belief_state[hypo] * likelihoods_for_answer[hypo]
                    for hypo in node.belief_state
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
                self._lookahead(evidence_node, curr_depth + 1)

    def _pose_question(self, question_node: QuestionNode) -> EvidenceNode:
        print("QUESTION".center(30, "="))
        print(question_node.question)
        print("ANSWERS".center(30, "="))
        for i, evidence_node in enumerate(question_node.children, start=1):
            print(f"{i}. {evidence_node.answer}")

        user_selection = int(input("Select an answer: "))
        return question_node.children[user_selection - 1]

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
