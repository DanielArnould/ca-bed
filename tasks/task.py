from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import string
from globals import SENTENCE_TRANSFORMER

from models import LLMRequestSession
from node import EvidenceNode, QuestionNode


class Task(ABC):
    """Abstract base class for all tasks that can be solved by UoT methods.
    Prompts and parsers are separated so that LLM calls remain independent of
    tasks and can be robustly tracked for history."""

    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    task_answer: str | None
    max_question_nodes: int  # Max number of questions to generate at each step
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float
    hypothesis_space: list[str]

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str | None,
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
    pattern = re.compile(r"(\d+)\.\s*(.+?)[\|;](.+?)(?=(?:\d+\.|$))", re.DOTALL)
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


def parse_binary_questions(output: str) -> list[Question]:
    pattern = re.compile(r"(\d+)\.\s*(.+?)(?=(?:\d+\.|$))", re.DOTALL)
    matches = pattern.findall(output)

    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.remove("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    questions = [
        Question(question=sanitise(q_text), possible_answers=["Yes", "No"])
        for _, q_text in matches
        if sanitise(q_text)
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


def parse_categorical_likelihoods(
    output: str, possible_answers: list[str]
) -> list[Likelihood]:
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.discard("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    # Work on a copy so we can safely pop answers as they're matched
    sanitised_possible_answers = [sanitise(a) for a in possible_answers]

    pattern = re.compile(r"^([^:]+):\s*(.*?)\s*Count", re.MULTILINE)
    matches = pattern.findall(output)

    if not matches:
        raise ValueError(
            f"No valid categorical likelihoods found in the output: {output}"
        )

    likelihoods_list: list[Likelihood] = []
    for answer_label, hypotheses_str in matches:
        sanitised_answer = sanitise(answer_label)

        # Step 1: Try exact match
        if sanitised_answer in sanitised_possible_answers:
            idx = sanitised_possible_answers.index(sanitised_answer)
        else:
            # Step 2: Fall back to semantic similarity
            answer_embeddings = SENTENCE_TRANSFORMER.encode(
                sanitised_possible_answers,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            query_embedding = SENTENCE_TRANSFORMER.encode(
                [sanitised_answer], convert_to_tensor=True, normalize_embeddings=True
            )
            similarities = SENTENCE_TRANSFORMER.similarity(
                query_embedding, answer_embeddings
            ).squeeze(0)
            idx = int(similarities.argmax().item())

        # Step 3: Remove matched answer to avoid duplicates
        sanitised_possible_answers.pop(idx)

        hypotheses = [sanitise(h) for h in hypotheses_str.split(",") if sanitise(h)]

        for hypothesis in hypotheses:
            # Build one-hot vector
            raw_vector = [
                1.0 if i == idx else 0.0 for i in range(len(possible_answers))
            ]

            # Clamp and normalise
            clamped = [max(min(val, 1 - 1e-10), 1e-10) for val in raw_vector]
            total = sum(clamped)
            normalised = [val / total for val in clamped] if total > 0 else clamped

            likelihoods_list.append(
                Likelihood(hypothesis=hypothesis, likelihoods=normalised)
            )

    if not likelihoods_list:
        raise ValueError(f"No valid hypotheses extracted from output: {output}")

    return likelihoods_list


@dataclass
class Probability:
    hypothesis: str
    probability: float


def parse_probabilities(output: str) -> list[Probability]:
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.discard("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    pattern = re.compile(r"\d+\.\s*([^|]+)\|([\d.]+)", re.MULTILINE)
    matches = pattern.findall(output)

    if not matches:
        raise ValueError(f"No valid probabilities found in the output: {output}")

    probabilities = []
    for hypothesis, prob_text in matches:
        sanitised_hypothesis = sanitise(hypothesis)
        probability = max(min(float(prob_text.strip()), 1 - 1e-10), 1e-10)
        probabilities.append(
            Probability(hypothesis=sanitised_hypothesis, probability=probability)
        )

    total = sum(p.probability for p in probabilities)
    if total > 0:
        for p in probabilities:
            p.probability /= total

    return probabilities


def parse_uniform_probabilities(output: str) -> list[Probability]:
    pattern = re.compile(r"(\d+)\.\s*(.+?)(?=(?:\d+\.|$))", re.DOTALL)
    matches = pattern.findall(output)

    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    allowed_chars.remove("|")

    def sanitise(text: str) -> str:
        return "".join(c for c in text if c in allowed_chars).strip()

    hypotheses = [sanitise(h) for _, h in matches if sanitise(h)]
    if not hypotheses:
        raise ValueError(f"No valid hypotheses found in the output: {output}")

    n = len(hypotheses)
    uniform_prob = 1.0 / n

    return [Probability(hypothesis=h, probability=uniform_prob) for h in hypotheses]


def parse_answer(output: str, question_node: QuestionNode) -> EvidenceNode:
    llm_answer = output.strip().lower()

    # First try exact match (case-insensitive)
    for child in question_node.children:
        if child.answer.strip().lower() == llm_answer:
            return child

    # Fall back to semantic similarity
    candidate_answers = [c.answer.strip() for c in question_node.children]
    answer_embeddings = SENTENCE_TRANSFORMER.encode(
        candidate_answers, convert_to_tensor=True, normalize_embeddings=True
    )
    output_embedding = SENTENCE_TRANSFORMER.encode(
        [llm_answer], convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = SENTENCE_TRANSFORMER.similarity(
        output_embedding, answer_embeddings
    ).squeeze(0)
    best_idx = int(similarities.argmax().item())
    best_child = question_node.children[best_idx]

    return best_child
