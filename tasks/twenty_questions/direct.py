import re
from textwrap import dedent
from typing import override

from node import EvidenceNode
from tasks.direct_prompting_task import DirectPromptingTask, NaiveQuestionerResponse


class Direct(DirectPromptingTask):
    self_report: str

    def __init__(
        self,
        task_answer: str,
        max_conversation_depth: int,
        hypothesis_space: list[str]
    ):
        self.task_answer = task_answer
        self.max_conversation_depth = max_conversation_depth
        self.hypothesis_space = hypothesis_space

    def __str__(self) -> str:
        return f"Twenty Questions (Bayesian): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=}"

    @override
    def get_questioner_prompt(self, current_node: EvidenceNode) -> str:
        history: list[tuple[str, str]] = []
        while current_node.parent:
            history.append((current_node.parent.question, current_node.answer))
            current_node = current_node.parent.parent

        history.reverse()
        hypothesis = "\n".join(f"- {hypo}" for hypo in self.hypothesis_space)

        base_prompt = dedent("""\
            You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X. X is possibly one of the following:
            {hypothesis}
            You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <This should ONLY be the exact name of the object from the list of possible objects>

            Otherwise, if you need more information, output:
            [QUESTION]: <Your question here>
        """)

        if not history:
            return base_prompt.format(
                hypothesis=hypothesis
            ).strip()

        conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        return (
            base_prompt.format(
                hypothesis=hypothesis
            )
            + "\n\nHere is the conversation history so far:\n"
            + conversation_history
            + (
                "\n\nNow you should make predictions instead of asking questions\n"
                if len(history) >= 14
                else ""
            )
        ).strip()

    @override
    def parse_questioner_output(
        self, output: str
    ) -> tuple[NaiveQuestionerResponse, str]:
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(
            r"\[(PREDICTION|ANSWER)\]:\s*(.*)", output, re.IGNORECASE
        )

        if question_match:
            return NaiveQuestionerResponse.QUESTION, question_match.group(1).strip()
        elif prediction_match:
            prediction = prediction_match.group(2).strip()
            return NaiveQuestionerResponse.PREDICTION, prediction
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    def get_answerer_prompt(self, question: str) -> str:
        return (
            dedent("""
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {target_item}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X.
            DO NOT REVEAL/MENTION WHAT X IS UNTIL I ASK "Is X ..."
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question)
            .strip()
        )
