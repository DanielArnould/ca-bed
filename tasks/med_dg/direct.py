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
from tasks.med_dg.data import MED_DG_SET, MedDGInstance


class Direct(DirectPromptingTask):
    instance: MedDGInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: MedDGInstance,
        max_conversation_depth: int,
    ):
        self.instance = instance
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=self.instance.disease,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=MED_DG_SET,
        )

    def __str__(self) -> str:
        return f"MedDG (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=} {self.instance.self_report=}"

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        parts = []

        # Prologue
        possible_diseases = "\n".join(f"- {h}" for h in self.hypothesis_space)
        parts.append(
            dedent(f"""\
            You are an expert medical doctor, and your patient self-reports that: {self.instance.self_report}. 
                   
            The patient is suffering from one of the following possible diseases:      
            {possible_diseases}
                   
            Your goal is to identify the correct disease.
            You can either ask a question to gather more information, or you can make a prediction.
            You must only ask questions that can be answered by only 'Yes' or 'No'
            
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <This should ONLY be the exact name of the disease from the list of possible diseases>

            Otherwise, if you need more information, output:
            [QUESTION]: <Your question here>
            """).strip()
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

        # Targetting prompt
        if len(history) >= self.max_conversation_depth - 3:
            parts.append(
                dedent("""
                Now you should make predictions instead of asking questions
                """).strip()
            )

        # Query LLM
        prompt = "\n\n".join(parts)
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
        prompt = dedent(f"""\
            You are a patient experiencing {self.instance.disease}. You self-reported that: {self.instance.self_report}.
            I am your doctor and I will ask you questions about your condition.  

            ### Instructions
            - Answer truthfully based on your symptoms.  
            - Review the available options before responding.  
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Question
            "{question}"
            """).strip()

        return await query_llm(prompt, self.answerer_session)
