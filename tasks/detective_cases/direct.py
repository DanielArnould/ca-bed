import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, get_conversation_history
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Prediction,
    Question,
)


class Direct(DirectPromptingTask):
    instance: DetectiveCasesInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: DetectiveCasesInstance,
        max_conversation_depth: int,
    ):
        self.instance = instance
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=next(
                suspect["name"]
                for suspect in self.instance["suspects"]
                if suspect.get("is_murderer", False)
            ),
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=[suspect["name"] for suspect in self.instance["suspects"]],
        )

    def __str__(self) -> str:
        return f"Detective Cases (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=}"

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        parts = []

        # Case background
        parts.append(
            dedent(f"""\
            You are a detective investigating a murder.  

            ### Case Background
            {self.get_case_background()}
            """).strip()
        )

        # Suspects info
        suspects_info_parts = []
        for idx, suspect in enumerate(self.instance["suspects"], start=1):
            suspects_info_parts.append(
                dedent(f"""\
                - Suspect {idx}:
                    - Name: {suspect["name"]}
                    - Introduction: {suspect["introduction"]}
                """).strip()
            )
        suspects_info = "\n".join(suspects_info_parts)
        parts.append(
            dedent(f"""\
            The investigation focuses on {len(self.hypothesis_space)} suspects:
            {suspects_info}
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

        # Instructions
        parts.append(
            dedent("""\
            ### Task
            Your goal is to identify the correct culprit.
            You can either ask a question to a specific suspect to gather more information,
            or you can make a prediction.
                   
            ### Response Format
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <Exact suspect name>
                   
            E.g., [PREDICTION]: Dr. Rose

            Otherwise, if you need more information, output:
            [QUESTION]: [Suspect Name] <Question text>
            
            E.g., [QUESTION]: [Professor Karpov] Where were you at 12:00PM?
            """)
        )

        # Targetting prompt
        if len(history) >= self.max_conversation_depth - 2:
            parts.append(
                dedent("""
                Now you should make predicitions instead of asking questions
                """).strip()
            )

        # Query LLM
        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(
            r"\[(PREDICTION|ANSWER|PREDECTION)\]:\s*(.*)", output, re.IGNORECASE
        )

        if question_match and self.parse_question(question_match.group(1).strip()):
            return Question(question_match.group(1).strip())
        elif prediction_match:
            return Prediction(prediction_match.group(2).strip())
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    async def query_answerer(self, question: str) -> str:
        suspect_name, actual_question = self.parse_question(question)

        suspect = next(
            (s for s in self.instance["suspects"] if s["name"] == suspect_name),
            None,
        )
        assert suspect is not None, f"Suspect '{suspect_name}' not found in case data"

        prompt = dedent(f"""\
            You are roleplaying as a suspect in a murder investigation.

            ### Suspect
            - Name: {suspect["name"]}
            - Task: {suspect["task"]}
            - Story: {suspect["story"]}

            ### Instructions
            - Answer the detective's question in character as {suspect_name}.
            - Stay consistent with your task and story.
            - You may lie, evade, or tell the truth depending on what seems natural for this suspect.
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.

            ### Detective's Question
            "{actual_question}"
        """)

        return await query_llm(prompt, self.answerer_session)

    def get_case_background(self) -> str:
        return dedent(f"""\
            Time: {self.instance["time"]}
            Location: {self.instance["location"]}
            Victim:
            - Name: {self.instance["victim"]["name"]}
            - Introduction: {self.instance["victim"]["introduction"]}
            - Cause of Death: {self.instance["victim"]["cause_of_death"]}
            - Murder Weapon: {self.instance["victim"]["murder_weapon"]}
        """)

    def parse_question(self, question: str) -> tuple[str, str]:
        suspect_match = re.match(r"\[(.*?)\]\s*(.*)", question)

        # Fallback to finding first name in string
        if not suspect_match:
            for name in self.hypothesis_space:
                if name in question:
                    parts = question.split(name, 1)
                    before, after = parts if len(parts) == 2 else ("", parts[0])
                    suspect_name = name
                    actual_question = (after or before).strip()
                    return suspect_name, actual_question

            assert False, f"Bad question: {question}"

        suspect_name, actual_question = suspect_match.groups()
        assert suspect_name in self.hypothesis_space, f"Unrecognised: {suspect_name}"
        return suspect_name, actual_question
