from abc import ABC, abstractmethod
from node import EvidenceNode, QuestionNode


class Task(ABC):
    """Abstract base class for all tasks that can be solved by UoT methods.
    Prompts and parsers are separated so that LLM calls remain independent of
    tasks and can be robustly tracked for history."""

    task_answer: str
    max_question_nodes: int  # Max number of questions to generate at each step
    max_evidence_nodes: int  # Max possible number of answers to each question
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float
    hypothesis_space: list[str]

    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_evidence_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        self.task_answer = task_answer
        self.max_question_nodes = max_question_nodes
        self.max_evidence_nodes = max_evidence_nodes
        self.max_lookahead_depth = max_lookahead_depth
        self.max_conversation_depth = max_conversation_depth
        self.confidence_threshold = confidence_threshold
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    def get_prior_prompt(self) -> str:
        pass

    @abstractmethod
    def parse_prior_output(self, output: str) -> dict[str, float]:
        pass

    @abstractmethod
    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        """
        From the current node, create a prompt that asks for new questions
        """
        pass

    @abstractmethod
    def parse_question_generation_output(self, output: str) -> list[str]:
        """Parse questions into question and possible answers"""
        pass

    @abstractmethod
    def get_likelihood_elicitation_prompt(self, question: str) -> str:
        """
        Given a question, create a prompt that asks for the likelihoods for every hypothesis for every answer
        """
        pass

    @abstractmethod
    def parse_likelihood_elicitation_output(
        self, output: str
    ) -> dict[str, dict[str, float]]:
        """
        For each answer of the question, return a dict mapping each hypothesis to the likelihood of the answer given the hypothesis.
        """
        pass

    @abstractmethod
    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        pass

    def parse_answer_selection_output(
        self, output: str, question_node: QuestionNode
    ) -> EvidenceNode:
        llm_answer = output.strip().lower()
        for child in question_node.children:
            if child.answer.lower() in llm_answer:
                return child

        raise RuntimeError(
            f"No matching answer selected for '{question_node.question}'. Possible answers: {list(child.answer for child in question_node.children)}, Given answer: {llm_answer}"
        )
