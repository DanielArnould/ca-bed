from abc import ABC, abstractmethod
from enum import Enum, auto

from node import EvidenceNode


class NaiveQuestionerResponse(Enum):
    PREDICTION = auto()
    QUESTION = auto()


class DirectPromptingTask(ABC):
    task_answer: str
    max_conversation_depth: int
    hypothesis_space: list[str]

    @abstractmethod
    def get_questioner_prompt(self, current_node: EvidenceNode) -> str:
        pass

    @abstractmethod
    def get_answerer_prompt(self, question: str) -> str:
        pass

    @abstractmethod
    def parse_questioner_output(
        self, output: str
    ) -> tuple[NaiveQuestionerResponse, str]:
        pass
