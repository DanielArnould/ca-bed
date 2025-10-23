import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, get_conversation_history
from tasks.direct_prompting_task import DirectPromptingTask, Question


class TwentyQuestionsDirect(DirectPromptingTask):
    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_conversation_depth: int,
        hypothesis_space: list[str],
    ):
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            "Twenty Questions (Direct): "
            f"{self.task_answer=} {self.max_conversation_depth=} "
            f"{self.hypothesis_space=}"
        )

    @override
    async def query_questioner(self, current_node: EvidenceNode) -> Question:
        parts = []

        # Prologue
        possible_items = "\n".join(f"- {h}" for h in self.hypothesis_space)
        parts.append(
            dedent(f"""\
            You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X. I will be impersonating the secret entity, X.
            The secret entity could be one of the following:
            {possible_items}

            You must always respond with exactly one yes/no question about X. Every question must:
            - Start with the words 'Is X'
            - Be answerable with 'Yes' or 'No'
            - Contain no additional commentary

            Respond using exactly this format on a single line:
            [QUESTION]: Is X <your question>?

            Do not answer outside of this question format.
            """)
            .format(possible_items=possible_items)
            .strip()
        )

        # Conversation history
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent(f"""\
                These are the questions you've already asked so far:
                {history_formatted}
                """).strip()
            )

        # Query LLM
        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)

        if question_match:
            question = question_match.group(1).strip()
            if not question.lower().startswith("is x"):
                raise RuntimeError(f"Question must start with 'Is X': {question}")
            return Question(question)

        raise RuntimeError(
            f"Response does not match expected structure. Output received: {output}"
        )

    @override
    async def query_answerer(self, question: str) -> str:
        prompt = dedent(f"""\
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {self.task_answer}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X.

            ### Instructions
            - Answer truthfully based on what X is.
            - Default to responding with exactly 'Yes' or 'No'.
            - If the question is explicitly asking whether X is {self.task_answer} (for example, "Is X {self.task_answer}?"), reply with 'Yes [CORRECT]' and nothing else.
            - Never append '[CORRECT]' to a response unless the question is a correct direct guess of X.
            - Do not add extra text or commentary.

            ### Question
            "{question}"
            """).strip()

        return await query_llm(prompt, self.answerer_session)

    def is_correct_signal(self, question: str, answer: str) -> bool:
        return "[CORRECT]" in answer.upper()
