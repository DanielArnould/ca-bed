import random
import re

from node import QuestionNode
from ..task import InteractionMode, Question, Task
from ...node import EvidenceNode


class Bayesian(Task):
    _secret_answer: str

    def __init__(self, max_question_nodes: int, interaction_mode: InteractionMode):
        super().__init__(
            interaction_mode, max_question_nodes, 2, ["Dog", "Cookie", "Paint", "Hat"]
        )
        if interaction_mode is InteractionMode.BENCHMARK:
            self._secret_answer = random.choice(self.hypothesis_space)

    def __str__(self) -> str:
        return f"Twenty Questions (Bayesian) {self.interaction_mode=}, {self._secret_answer=}, {self.max_question_nodes=}, {self.hypothesis_space=}"

    def get_initial_belief_state(self) -> dict[str, float]:
        prob = 1.0 / len(self.hypothesis_space)
        return {item: prob for item in self.hypothesis_space}

    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        formatted_belief_state = "\n".join(
            f"- {hypothesis}: {prob:.0%}"
            for hypothesis, prob in current_node.belief_state
        )

        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append(f"Q: {question}\nA: {answer}")
            node = node.parent.parent

        formatted_history = "\n".join(reversed(history))
        prompt = f"""
        You are an expert player in a 20-questions game.
        Your goal is to ask questions that will best help you distinguish between the possible answers.

        Here is the current probability distribution (belief state) over the possible answers:
        {formatted_belief_state}

        Here is the conversation history so far:
        {formatted_history}

        Based on the current beliefs and history, what are the best (maximum {self.max_question_nodes}) Yes/No questions to ask next?
        Return them as a numbered list. For example:
        1. Does it have fur?
        2. Can it fly?
        """

        return prompt.strip()

    def parse_question_generation_output(self, output: str) -> list[Question]:
        question_texts: list[str] = re.findall(r"^\d+\.\s*(.*)", output, re.MULTILINE)
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
        assert current_node.parent is not None, "current node cannot be the root!"

        formatted_hypotheses = ", ".join(
            f'"{hypothesis}"' for hypothesis in self.hypothesis_space
        )

        prompt = f""""
        Consider the question: "{question.question}"

        For each possible answer ("Yes", "No") and for each possible thing ({formatted_hypotheses}), estimate the probability P(Answer | Thing).
        Provide the output as a structured list. For example:

        Answer: Yes
        Dog: 0.9
        Cat: 0.8
        Bird: 0.1

        Answer: No
        Dog: 0.1
        Cat: 0.2
        Bird: 0.9
        """

        return prompt.strip()

    def parse_likelihood_elicitation_output(
        self, output: str, question: Question
    ) -> dict[str, dict[str, float]]:
        likelihoods: dict[str, dict[str, float]] = {
            answer: {} for answer in question.answers
        }

        # Split the output into sections for each answer
        answer_sections = re.split(r"Answer:\s*", output, flags=re.IGNORECASE)

        for section in answer_sections:
            if not section.strip():
                continue

            # The first line should be the answer (e.g., "Yes")
            lines = section.strip().split("\n")
            current_answer = lines[0].strip()

            if current_answer not in question.answers:
                continue

            # Parse the likelihoods for each hypothesis
            for line in lines[1:]:
                match = re.match(r"([^:]+):\s*((0|1).\d+)", line)
                assert match is not None, "Line is unrecognised"
                hypothesis: str = match.group(1).strip()
                prob = float(match.group(2))
                assert hypothesis in self.hypothesis_space, "Hypothesis is unrecognised"
                assert 0.0 <= prob <= 1.0, "Likelihood is not a probability!"
                likelihoods[current_answer][hypothesis] = prob

        assert len(likelihoods) == len(question.answers), "Not all answers considered!"
        return likelihoods

    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        prompt = f"""
        You are playing a game of 20 questions.
        The secret thing you are thinking of is a {self._secret_answer}.
        The question you have been asked is: "{question_node.question}"

        What is the correct answer, "Yes" or "No"?
        Please respond with only the word "Yes" or "No".
        """
        return prompt.strip()

    def parse_answer_selection_output(
        self, output: str, question_node: QuestionNode
    ) -> EvidenceNode:
        llm_answer = output.strip().lower()
        evidence_node = next(
            (
                child
                for child in question_node.children
                if child.answer.lower() == llm_answer
            ),
            None,
        )
        assert evidence_node is not None, "No matching answer selected"
        return evidence_node


class Baseline: ...


class Nonbinary: ...


class NonbinaryBayesian: ...
