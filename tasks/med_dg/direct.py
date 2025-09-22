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
        hypothesis_space: list[str],
        self_report: str,
    ):
        self.self_report = self_report
        self.task_answer = task_answer
        self.max_conversation_depth = max_conversation_depth
        self.hypothesis_space = hypothesis_space

    def __str__(self) -> str:
        return f"MedDG (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=} {self.self_report=}"

    @override
    def get_questioner_prompt(self, current_node: EvidenceNode) -> str:
        history: list[tuple[str, str]] = []
        while current_node.parent:
            history.append((current_node.parent.question, current_node.answer))
            current_node = current_node.parent.parent

        history.reverse()
        possible_diseases = "\n".join(f"- {hypo}" for hypo in self.hypothesis_space)

        base_prompt = dedent("""\
            You are an expert medical doctor. Your patient self-reports that: {self_report}. 
               
            The patient is suffering from one of the following possible diseases:      
            {possible_diseases}
    
            Your goal is to identify the correct disease. You can either ask a yes/no question to gather more information, or you can make a prediction.
            
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <This should ONLY be the exact name of the disease from the list of possible diseases>

            Otherwise, if you need more information, output:
            [QUESTION]: <Your question here>
            """)

        if history:
            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
            return (
                base_prompt.format(
                    self_report=self.self_report, possible_diseases=possible_diseases
                )
                + "\n\nHere is the conversation history so far:\n"
                + conversation_history
            ).strip()

        return base_prompt.format(
            self_report=self.self_report, possible_diseases=possible_diseases
        ).strip()

    @override
    def parse_questioner_output(
        self, output: str
    ) -> tuple[NaiveQuestionerResponse, str]:
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(r"\[PREDICTION\]:\s*(.*)", output, re.IGNORECASE)

        if question_match:
            return NaiveQuestionerResponse.QUESTION, question_match.group(1).strip()
        elif prediction_match:
            prediction = prediction_match.group(1).strip()
            return NaiveQuestionerResponse.PREDICTION, prediction
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    def get_answerer_prompt(self, question: str) -> str:
        return (
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you questions, and you should answer each one truthfully based on your disease.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question)
            .strip()
        )
