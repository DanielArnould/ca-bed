import json
import random
import re

from method import get_conversation_depth
from node import EvidenceNode, QuestionNode
from .data import MED_DG_SET
import tasks.med_dg.bayesian_prompts as bayesian
import tasks.med_dg.baseline_prompts as baseline
from .data import load_data
from ..task import Question, Task

class Bayesian(Task):
    def __init__(self, max_question_nodes: int):
        hypothesis_space = MED_DG_SET
        med_dg_data = load_data()
        chosen_patient = random.choice(med_dg_data)
        self.task_answer = chosen_patient['target']
        self.problem_desc = chosen_patient['self_repo']
        super().__init__(
            self.task_answer,
            max_question_nodes,
            max_evidence_nodes=2,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Bayesian): Answer: {self.task_answer}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

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

        prologue = bayesian.get_questioner_prologue(problem_description=self.problem_desc, hypothesis_space=self.hypothesis_space)
        generation_prompt = bayesian.get_question_generation_prompt(
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
        return bayesian.get_verbalization_probability_elicitation_prompt(
            problem_description=self.problem_desc, hypothesis_space=self.hypothesis_space, question=question.question
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
        return bayesian.get_answer_selection_prompt(
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


class Baseline(Task):
    def __init__(self, max_question_nodes: int):
        hypothesis_space = MED_DG_SET
        med_dg_data = load_data()
        chosen_patient = random.choice(med_dg_data)
        self.task_answer = chosen_patient['target']
        self.problem_desc = chosen_patient['self_repo']

        super().__init__(
            self.task_answer,
            max_question_nodes,
            max_evidence_nodes=2,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG Questions (Non-Bayesian): Answer: {self.task_answer}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

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

        prologue = baseline.get_questioner_prologue(problem_description=self.problem_desc)
        generation_prompt = baseline.get_question_generation_prompt(
            m=self.max_question_nodes,
            history=history,
            belief_state=list(current_node.belief_state.items()),
        )
        target_prompt = ""
        if get_conversation_depth(current_node) >= 14:
            target_prompt = baseline.get_targeting_prompt()

        return f"{prologue}\n\n{generation_prompt}\n\n{target_prompt}"

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
        return baseline.get_verbalization_probability_elicitation_prompt(
            problem_description=self.problem_desc,
            belief_state=list(current_node.belief_state.items()),
            question=question.question,
        )

    def parse_likelihood_elicitation_output(
        self, output: str, question
    ) -> dict[str, dict[str, float]]:
        # grab text after YES: until "Count"
        m_yes = re.search(r"YES:\s*(.*?)\s*Count", output, re.DOTALL)
        m_no = re.search(r"NO:\s*(.*?)\s*Count", output, re.DOTALL)

        yes_set = set([s.strip() for s in m_yes.group(1).split(",")] if m_yes else [])
        no_set = set([s.strip() for s in m_no.group(1).split(",")] if m_no else [])
        hs = set(self.hypothesis_space)
        likelihoods_yes = {h: (1.0 if h in yes_set else 0.0) for h in hs}
        likelihoods_no = {h: (1.0 if h in no_set else 0.0) for h in hs}

        return {"Yes": likelihoods_yes, "No": likelihoods_no}

    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        return baseline.get_answer_selection_prompt(
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