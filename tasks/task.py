from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import string
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
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        self.questioner_session = questioner_session
        self.answerer_session = answerer_session
        self.task_answer = task_answer
        self.max_question_nodes = max_question_nodes
        self.max_lookahead_depth = max_lookahead_depth
        self.max_conversation_depth = max_conversation_depth
        self.confidence_threshold = confidence_threshold
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    async def create_initial_belief_state(self) -> dict[str, float]:
        pass

    @abstractmethod
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        pass

    @abstractmethod
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        pass

    @abstractmethod
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        pass

    # TODO: Create __str__


@dataclass
class Question:
    question: str
    possible_answers: list[str]


def parse_questions(output: str) -> list[Question]:
    pattern = re.compile(r"(\d+)\.\s*(.+?)\|(.+?)(?=(?:\d+\.|$))", re.DOTALL)
    matches = pattern.findall(output)

    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.remove("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    questions = []
    for _, question_text, answers_text in matches:
        question = question_text.strip()
        possible_answers = [sanitise(a) for a in answers_text.split(";") if sanitise(a)]
        if question and possible_answers:
            questions.append(
                Question(question=question, possible_answers=possible_answers)
            )

    if not questions:
        raise ValueError(f"No valid questions found in the output: {output}")

    return questions


def parse_questions_only(output: str) -> list[Question]:
    pattern = re.compile(r"(\d+)\.\s*(.+?)(?=(?:\d+\.|$))", re.DOTALL)
    matches = pattern.findall(output)

    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.discard("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    questions = [
        Question(question=sanitise(q_text), possible_answers=["Yes", "No"])
        for _, q_text in matches
    ]

    if not questions:
        raise ValueError(f"No valid questions found in the output: {output}")

    return questions


@dataclass
class Likelihood:
    hypothesis: str
    likelihoods: list[float]


def parse_likelihoods(output: str) -> list[Likelihood]:
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.remove("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    pattern = re.compile(r"\d+\.\s*([^|]+)\|([\d.;]+)", re.MULTILINE)
    matches = pattern.findall(output)

    if not matches:
        raise ValueError(f"No valid likelihoods found in the output: {output}")

    likelihoods_list = []
    for hypothesis, probs_text in matches:
        sanitised_hypothesis = sanitise(hypothesis)
        probs = [
            max(min(float(p.strip()), 1 - 1e-10), 1e-10)
            for p in probs_text.split(";")
            if p.strip()
        ]
        total = sum(probs)
        normalised_probs = [p / total for p in probs] if total > 0 else probs
        likelihoods_list.append(
            Likelihood(hypothesis=sanitised_hypothesis, likelihoods=normalised_probs)
        )

    return likelihoods_list
