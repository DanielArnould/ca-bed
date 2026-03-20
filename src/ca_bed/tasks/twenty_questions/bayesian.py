import re
from textwrap import dedent
from typing import override


from ca_bed.llm import get_response
from ca_bed.node import (
    QuestionAnswer,
)
from ca_bed.tasks.task import TreeBasedTask


class TwentyQuestionsBayesian(TreeBasedTask):
    @override
    def get_id(self) -> str:
        return f"20q_bayesian_{self.task_answer}"

    @override
    async def create_questions(
        self, conversation_history: list[QuestionAnswer], belief_state: dict[str, float]
    ) -> dict[str, list[str]]:
        possible_entities = [
            entity for entity, prob in belief_state.items() if prob > 0
        ]
        prompt = build_question_generation_prompt(
            conversation_history, possible_entities, self.n_questions
        )

        response = await get_response(prompt, self.questioner_llm)
        questions = parse_question_generation_response(response)
        return {q: ["Yes", "No"] for q in questions[: self.n_questions]}

    @override
    async def get_likelihoods(
        self, question: str, possible_answers: list[str]
    ) -> dict[str, dict[str, float]]:
        prompt = build_likelihood_prompt(question, self.hypothesis_space)

        response = await get_response(prompt, self.questioner_llm)
        likelihoods = parse_likelihood_response(response)
        return {
            "Yes": {entity: p for entity, p in likelihoods.items()},
            "No": {entity: 1 - p for entity, p in likelihoods.items()},
        }

    @override
    async def get_answer(self, question: str, possible_answers: list[str]) -> str:
        prompt = build_answer_prompt(self.task_answer, question)
        response = await get_response(prompt, self.answerer_llm)
        answer = parse_answer_response(response)
        return answer


def build_question_generation_prompt(
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    n_questions: int,
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
