import re
from textwrap import dedent
from typing import override

from ca_bed.llm import get_response
from ca_bed.node import QuestionAnswer
from ca_bed.tasks.task import DirectTask, Prediction, Question


class TwentyQuestionsDirect(DirectTask):
    @override
    def get_id(self) -> str:
        return f"20q_direct_{self.task_answer}"

    @override
    async def query_questioner(
        self, conversation_history: list[QuestionAnswer]
    ) -> Question | Prediction:
        prompt = build_questioner_prompt(
            conversation_history, self.hypothesis_space, self.max_conversation_depth
        )

        response = await get_response(prompt, self.questioner_llm)
        return parse_questioner_response(response)

    @override
    async def query_answerer(self, question: str) -> str:
        prompt = build_answerer_prompt(self.task_answer, question)

        response = await get_response(prompt, self.answerer_llm)
        return parse_answerer_response(response)


def build_questioner_prompt(
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    max_conversation_depth: int,
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

    if len(conversation_history) >= max_conversation_depth - 3:
        prompt_parts.append(
            "You are running out of questions. You must now make a prediction instead of asking a question."
        )
    else:
        prompt_parts.append(
            "You can either ask an excellent yes/no question that you could ask next, or make a prediction of what X is."
        )

    prompt_parts.extend(
        [
            "First, provide a short explanation for your reasoning. Then provide the result strictly in one of the following formats, including the double hashtag:",
            "",
            "If asking a question:",
            "##Question##: <One question here>",
            "",
            "If making a prediction:",
            "##Prediction##: <The exact name of the entity here>",
        ]
    )

    return "\n".join(prompt_parts)


def parse_questioner_response(response: str) -> Question | Prediction:
    question_regex = r"##Question##[^#]*:\s*(.*)"
    prediction_regex = r"##Prediction##[^#]*:\s*(.*)"

    question_match = re.search(question_regex, response, re.IGNORECASE)
    prediction_match = re.search(prediction_regex, response, re.IGNORECASE)

    if question_match:
        return Question(question_match.group(1).strip())
    elif prediction_match:
        return Prediction(prediction_match.group(1).strip())
    else:
        raise RuntimeError(
            f"Response does not match expected structure in '{response}'"
        )


def build_answerer_prompt(secret_entity: str, question: str) -> str:
    return dedent(f"""\
        You are a player of the 20 Questions game. Your goal is
        to impersonate the secret entity, X. X is {secret_entity}. You have just
        been asked, '{question}'. Answer truthfully based on what X is, then
        provide the result strictly in the format, including the double hashtag:

        ##Answer##: <'Yes' or 'No'>
    """).strip()


def parse_answerer_response(response: str) -> str:
    answer_regex = r"##Answer##[^#]*:\s*(Yes|No)"
    matches = re.findall(answer_regex, response, re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(f"Unexpected number of matches in '{response}'")
    return matches[0].capitalize()
