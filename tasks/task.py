from abc import ABC, abstractmethod
from ..node import AnswerNode, QuestionGroup, Item


class Task(ABC):
    """Abstract base class for all tasks that can be solved by UoT methods.
    Prompts and parsers are separated so that LLM calls remain independent of
    tasks and can be robustly tracked for history."""

    @abstractmethod
    def create_root(self) -> AnswerNode:
        """Creates the initial root node for the task's search tree with the
        possible answer set."""
        pass

    @abstractmethod
    def create_binary_questions_prompt(
        self, current_node: AnswerNode, question_history: list[str]
    ) -> str:
        """Creates the prompt to ask an LLM to generate a new binary question."""
        pass

    @abstractmethod
    def create_binary_questions(
        self, llm_response: str, parent_node: AnswerNode
    ) -> list[QuestionGroup]:
        """Parses the LLM's response and creates the corresponding QuestionGroup."""
        pass

    @abstractmethod
    def answerer_prologue(self, answer: Item) -> str:
        """Creates the system prompt for the 'answerer' LLM."""
        pass

    @abstractmethod
    def create_answerer_prompt(
        self, prologue: str, question: str, history: list[tuple[str, str]]
    ) -> str:
        """Creates the prompt to ask the 'answerer' LLM a question."""
        pass

    @abstractmethod
    def parse_answerer_response(
        self, question_group: QuestionGroup, llm_response: str
    ) -> AnswerNode | None:
        """Parses the answerer's response to select the next node in the tree."""
        pass
