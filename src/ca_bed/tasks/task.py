from abc import ABC, abstractmethod

from ca_bed.llm import LLM
from ca_bed.node import EvidenceNode, ProbabilityDistribution, QuestionNode


class Task(ABC):
    @abstractmethod
    def get_id(self) -> str: ...

    @abstractmethod
    def get_expected_answer(self) -> str: ...

    @abstractmethod
    async def create_initial_belief_state(self) -> ProbabilityDistribution: ...

    @abstractmethod
    async def create_questions(
        self, curr: EvidenceNode, n_questions: int, llm: LLM
    ) -> dict[str, list[str]]: ...

    @abstractmethod
    async def get_likelihoods(
        self, curr: QuestionNode, llm: LLM
    ) -> dict[str, dict[str, float]]: ...

    @abstractmethod
    async def get_answer(self, curr: QuestionNode, llm: LLM) -> str: ...
