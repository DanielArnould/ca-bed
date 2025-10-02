import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, get_conversation_history
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Prediction,
    Question,
)


class Direct(DirectPromptingTask):
    self_report: str

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_conversation_depth: int,
        hypothesis_space: list[str],
        self_report: str,
    ):
        self.self_report = self_report
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=} {self.self_report=}"

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        parts = []

        # Prologue
        possible_diseases = "\n".join(f"- {h}" for h in self.hypothesis_space)
        parts.append(
            dedent("""\
            You are an expert medical doctor, and your patient self-reports that: {self_report}. 
                   
            The patient is suffering from one of the following possible diseases:      
            {possible_diseases}
                   
            Your goal is to identify the correct disease.
            You can either ask a question to gather more information, or you can make a prediction.
            
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <This should ONLY be the exact name of the disease from the list of possible diseases>

            Otherwise, if you need more information, output:
            [QUESTION]: <Your question here>
            """)
            .format(self_report=self.self_report, possible_diseases=possible_diseases)
            .strip()
        )

        # Conversation History
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent("""
                These are the questions you've asked to the patient so far:
                {history}
                """)
                .format(history=history_formatted)
                .strip()
            )

        # Targetting prompt
        if len(history) >= 0.7 * self.max_conversation_depth:
            parts.append(
                dedent("""
                Now you should make predicitions instead of asking questions
                """).strip()
            )

        # Query LLM
        prompt = "\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(
            r"\[(PREDICTION|ANSWER)\]:\s*(.*)", output, re.IGNORECASE
        )

        if question_match:
            return Question(question_match.group(1).strip())
        elif prediction_match:
            return Prediction(prediction_match.group(2).strip())
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    async def query_answerer(self, question: str) -> str:
        prompt = (
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you questions, and you should answer each one truthfully based on your disease.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question)
            .strip()
        )

        return await query_llm(prompt, self.answerer_session, max_tokens=50)
