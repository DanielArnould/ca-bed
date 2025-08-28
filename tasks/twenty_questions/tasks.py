import random
import re
import json

from node import QuestionNode, EvidenceNode
from tasks.twenty_questions.data import THING200
from tasks.twenty_questions.prompts import (
    get_answer_selection_prompt,
    get_question_generation_prompt,
    get_questioner_prologue,
    get_verbalization_probability_elicitation_prompt,
)
from ..task import Question, Task


class Bayesian(Task):
    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        super().__init__(
            task_answer,
            max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"Twenty Questions (Bayesian): Answer: {self.task_answer}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

    def get_initial_belief_state(self) -> dict[str, float]:
        prob = 1.0 / len(self.hypothesis_space)
        return {item: prob for item in self.hypothesis_space}

    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append((question, answer))
            node = node.parent.parent

        prologue = get_questioner_prologue(hypothesis_space=self.hypothesis_space)
        generation_prompt = get_question_generation_prompt(
            m=self.max_question_nodes,
            history=history,
            belief_state=list(current_node.belief_state.items()),
        )

        return f"{prologue}\n\n{generation_prompt}"

    def parse_question_generation_output(self, output: str) -> list[Question]:
        question_texts: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )
        questions = []

        for question_text in question_texts:
            clean_question_text = question_text.strip().replace("?", "") + "?"
            questions.append(
                Question(question=clean_question_text, answers=["Yes", "No"])
            )

        return questions

    def get_likelihood_elicitation_prompt(
        self, current_node: EvidenceNode, question: Question
    ) -> str:
        return get_verbalization_probability_elicitation_prompt(
            hypothesis_space=self.hypothesis_space, question=question.question
        )

    def parse_likelihood_elicitation_output(
        self, output: str, question: Question
    ) -> dict[str, dict[str, float]]:
        likelihoods: dict[str, dict[str, float]] = {
            answer: {} for answer in question.answers
        }
        cleaned_output = output[output.find("{") : output.rfind("}") + 1]
        probs = json.loads(cleaned_output)
        likelihoods["Yes"] = probs
        likelihoods["No"] = {item: 1 - prob for item, prob in probs.items()}
        return likelihoods

    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        return get_answer_selection_prompt(
            ground_truth=self.task_answer, question=question_node.question
        )

    def parse_answer_selection_output(
        self, output: str, question_node: QuestionNode
    ) -> EvidenceNode:
        llm_answer = output.strip().lower()
        for child in question_node.children:
            if child.answer.lower() in llm_answer:
                return child

        assert False, (
            f"No matching answer selected. Possible answers: {list(child.answer for child in question_node.children)} Actual answer: {llm_answer}"
        )


class Baseline: ...
