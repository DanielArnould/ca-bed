import re
from textwrap import dedent
from typing import override


from ca_bed.llm import LLM, get_response
from ca_bed.node import (
    EvidenceNode,
    ProbabilityDistribution,
    QuestionAnswer,
    QuestionNode,
    get_conversation_history,
)
from ca_bed.tasks.task import Task


class TwentyQuestionsBayesian(Task):
    def __init__(
        self,
        secret_entity: str,
        entities: list[str],
    ):
        self.secret_entity = secret_entity
        self.entities = entities

    @override
    def get_id(self) -> str:
        return f"20q_bayesian_{self.secret_entity}"

    @override
    def get_expected_answer(self) -> str:
        return self.secret_entity

    @override
    async def create_initial_belief_state(self) -> ProbabilityDistribution:
        return {entity: 1.0 / len(self.entities) for entity in self.entities}

    @override
    async def create_questions(
        self, curr: EvidenceNode, n_questions: int, llm: LLM
    ) -> dict[str, list[str]]:
        possible_entities = [
            entity for entity, prob in curr.belief_state.items() if prob > 0
        ]
        conversation_history = get_conversation_history(curr)
        prompt = build_question_generation_prompt(
            conversation_history, possible_entities, n_questions
        )

        response = await get_response(prompt, llm, self.get_id())
        questions = parse_question_generation_response(response)
        return {q: ["Yes", "No"] for q in questions[:n_questions]}

    @override
    async def get_likelihoods(
        self, curr: QuestionNode, llm: LLM
    ) -> dict[str, dict[str, float]]:
        prompt = build_likelihood_prompt(curr.question, self.entities)

        response = await get_response(prompt, llm, self.get_id())
        likelihoods = parse_likelihood_response(response)
        return {
            "Yes": {entity: p for entity, p in likelihoods.items()},
            "No": {entity: 1 - p for entity, p in likelihoods.items()},
        }

    @override
    async def get_answer(self, curr: QuestionNode, llm: LLM) -> str:
        prompt = build_answer_prompt(self.secret_entity, curr.question)
        response = await get_response(prompt, llm, self.get_id())
        answer = parse_answer_response(response)
        return answer


def build_question_generation_prompt(
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    n_questions: int = 3,
) -> str:
    possible_entities_str = "; ".join(entities)
    prompt_parts = [
        "You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X.",
        f"The secret entity X is one of these: {possible_entities_str}.",
    ]

    if conversation_history:
        prompt_parts.append(" The game so far:")
        prompt_parts.extend(
            f"{idx}. Q: {qa.question}; A: {qa.answer}"
            for idx, qa in enumerate(conversation_history, start=1)
        )

    is_plural = n_questions > 1
    prompt_parts.extend(
        [
            f"What {'are' if is_plural else 'is'} {n_questions} excellent yes/no question{'s' if is_plural else ''} that you could ask next?",
            "Provide a short explanation for your reasoning. Then provide the result strictly in the format, including the double hashtag: ",
            "##Question##: <One question here>",
            "##Question##: <One question here>",
        ]
    )
    return "\n".join(prompt_parts)


def parse_question_generation_response(response: str) -> list[str]:
    question_generation_regex = r"##Question##[^#]*:[^ ]* ((?:.*)\?)"
    return re.findall(question_generation_regex, response)


def build_likelihood_prompt(question: str, entities: list[str]) -> str:
    possible_entities_str = "; ".join(entities)

    return dedent(f"""\
            You are analysing a game of 20 questions. The question
            asked was "{question}". The possible answers are "Yes" and "No". The
            possible entities are: {possible_entities_str}. For each entity,
            assuming it was the answer, how likely is it that the answerer would
            answer yes? First, provide a short explanation of your reasoning.
            Then provide the result strictly in the following format, including the double
            hashtag.

            ##Apple##: <a single number between 0 and 1>
            ##Bee##: <a single number between 0 and 1>""").strip()


def parse_likelihood_response(response: str) -> dict[str, float]:
    likelihood_regex = r"##(.*)##[^#]*:[^ ]* (\d+(?:\.\d+)?)"
    return {
        key: float(likelihood)
        for key, likelihood in re.findall(likelihood_regex, response)
    }


def build_answer_prompt(secret_entity: str, question: str) -> str:
    return dedent(f"""\
        You are a player of the 20 Questions game. Your goal is
        to impersonate the secret entity, X. X is {secret_entity}. You have just
        been asked, '{question}'. Answer truthfully based on what X is, then
        provide the result strictly in the format, including the double hashtag:

        ##Answer##: <'Yes' or 'No'>
    """).strip()


def parse_answer_response(response: str) -> str:
    answer_regex = r"##Answer##[^#]*:[^ ]* (Yes|No)"
    matches = re.findall(answer_regex, response)
    if len(matches) != 1:
        raise ValueError(f"Unexpected number of matches in '{response}'")
    return matches[0]
