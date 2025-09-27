from abc import ABC, abstractmethod
from models import LLMRequestSession
from node import EvidenceNode, QuestionNode


class Task(ABC):
    """Abstract base class for all tasks that can be solved by UoT methods.
    Prompts and parsers are separated so that LLM calls remain independent of
    tasks and can be robustly tracked for history."""

    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    task_answer: str
    max_question_nodes: int  # Max number of questions to generate at each step
    max_evidence_nodes: int  # Max possible number of answers to each question
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float
    hypothesis_space: list[str]

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_evidence_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        self.questioner_session = questioner_session
        self.answerer_session = answerer_session
        self.task_answer = task_answer
        self.max_question_nodes = max_question_nodes
        self.max_evidence_nodes = max_evidence_nodes
        self.max_lookahead_depth = max_lookahead_depth
        self.max_conversation_depth = max_conversation_depth
        self.confidence_threshold = confidence_threshold
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    async def create_initial_belief_state(self) -> dict[str, float]:
        pass

    @abstractmethod
    async def create_questions(self, current_node: EvidenceNode) -> list[str]:
        pass

    @abstractmethod
    async def get_likelihoods(
        self, current_node: QuestionNode
    ) -> dict[str, dict[str, float]]:
        pass

    @abstractmethod
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        pass
