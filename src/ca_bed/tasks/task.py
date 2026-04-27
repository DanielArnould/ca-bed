from abc import ABC, abstractmethod
from dataclasses import dataclass

from ca_bed.llm import LLM
from ca_bed.node import ProbabilityDistribution, QuestionAnswer


class Task(ABC):
    def __init__(
        self,
        questioner_llm: LLM,
        answerer_llm: LLM,
        task_answer: str,
        hypothesis_space: list[str],
        max_conversation_depth: int,
    ) -> None:
        self.questioner_llm = questioner_llm
        self.answerer_llm = answerer_llm
        self.task_answer = task_answer
        self.hypothesis_space = hypothesis_space
        self.max_conversation_depth = max_conversation_depth

    @abstractmethod
    def get_id(self) -> str: ...

    def get_task_answer(self) -> str:
        return self.task_answer


class TreeBasedTask(Task):
    def __init__(
        self,
        questioner_llm: LLM,
        answerer_llm: LLM,
        task_answer: str,
        hypothesis_space: list[str],
        max_conversation_depth: int,
        n_questions: int,
        max_lookahead_depth: int,
        confidence_threshold: float,
        estimator_confidence: float,
    ) -> None:
        super().__init__(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=task_answer,
            hypothesis_space=hypothesis_space,
            max_conversation_depth=max_conversation_depth,
        )
        self.n_questions = n_questions
        self.max_lookahead_depth = max_lookahead_depth
        self.confidence_threshold = confidence_threshold
        self.estimator_confidence = estimator_confidence

    def create_uniform_belief_state(self) -> ProbabilityDistribution:
        return {
            hypothesis: 1 / len(self.hypothesis_space)
            for hypothesis in self.hypothesis_space
        }

    @abstractmethod
    async def create_questions(
        self, conversation_history: list[QuestionAnswer], belief_state: dict[str, float]
    ) -> dict[str, list[str]]: ...

    @abstractmethod
    async def get_likelihoods(
        self, question: str, possible_answers: list[str]
    ) -> dict[str, dict[str, float]]: ...

    @abstractmethod
    async def get_answer(self, question: str, possible_answers: list[str]) -> str: ...


@dataclass(slots=True, frozen=True)
class Prediction:
    value: str


@dataclass(slots=True, frozen=True)
class Question:
    value: str


class DirectTask(Task):
    @abstractmethod
    async def query_questioner(
        self, conversation_history: list[QuestionAnswer]
    ) -> Question | Prediction: ...

    @abstractmethod
    async def query_answerer(self, question: str) -> str: ...
