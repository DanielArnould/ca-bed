import re
from textwrap import dedent
from typing import override

from tasks.task import DPTask


class Direct(DPTask):
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
    def get_questioner_prompt(self, history: list[tuple[str, str]]) -> str:
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
            
            The questions should be answerable with 'Yes' or 'No'.
            """)
        
        if history:
            conversation_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
            return (
                base_prompt.format(self_report=self.self_report, possible_diseases=possible_diseases) +
                "\n\nHere is the conversation history so far:\n" +
                conversation_history
            ).strip()

        return base_prompt.format(self_report=self.self_report, possible_diseases=possible_diseases).strip()

    @override
    def parse_questioner_output(self, output: str) -> tuple[bool, str]:
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(r"\[PREDICTION\]:\s*(.*)", output, re.IGNORECASE)

        if question_match:
            return False, question_match.group(1).strip()
        elif prediction_match:
            prediction = prediction_match.group(1).strip()

            assert any(prediction.lower() == label.strip().lower() for label in self.hypothesis_space), RuntimeError(f'{prediction} is not part of the hypothesis space')
            return True, prediction
        else:
            # Default to interpreting the output as a question if no tag is found
            return False, output.strip()

    @override
    def get_answerer_prompt(self, question: str) -> str:
        return (
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you questions, and you should answer each one truthfully based on your disease, by saying 'Yes' or 'No'. 
            ONLY ANSWER WITH YES OR NO.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question)
            .strip()
        )

    @override
    def parse_answerer_output(self, output: str) -> str:
        return output.strip()
