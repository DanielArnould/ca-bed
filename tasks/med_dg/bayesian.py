import json
import re
from textwrap import dedent
from typing import override

from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from tasks.task import Task


class Bayesian(Task):
    self_report: str

    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
        self_report: str,
    ):
        self.self_report = self_report
        super().__init__(
            task_answer,
            max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=} {self.self_report=}"

    @override
    async def create_root(self, model: Model) -> tuple[EvidenceNode, int, int]:
        prompt = (
            dedent("""\
            You are an expert Doctor, you are given this self-report:
            {self_report}
            
            Below is the set of possible conditions the patient may have:
            {hypothesis_space}
            
            Given your knowledge in this area and the given self-report, please assign a prior belief to each 
            item in the hypothesis set provided above, nothing else. 
            
            For example, if our hypothesis space is [A,B,C,D], and C and A are more relevant given the self-report, you 
            might assign probabilities like below: 
            A: 0.35, B: 0.02, C: 0.55, D: 0.08
            
            Note the probabilities are just conjectures, you should generate reasonable probabilities for each option. 
            Also note that the sum of these prior probabilities is 1. Please give a probability for each hypothesis even 
            if they are extremely small.
            
            Please strictly return your response in the format below:
            {{
                "<Hypothesis 1 full name>": <probability>, 
                "<Hypothesis 2 full name>": <probability>, 
                ....
            }}    
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=self.hypothesis_space,
            )
            .strip()
        )

        output, input_tokens, output_tokens = await call_llm(prompt, model)
        cleaned_output = output[output.rfind("{") : output.rfind("}") + 1]
        probs: dict = json.loads(cleaned_output)
        adjusted_prior = {h: probs.get(h, 1e-10) for h in self.hypothesis_space}
        total = sum(adjusted_prior.values())
        adjusted_prior = {h: v / total for h, v in adjusted_prior.items()}

        return EvidenceNode("ROOT", adjusted_prior, 1.0), input_tokens, output_tokens

    @override
    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        prompt_parts = []
        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append((question, answer))
            node = node.parent.parent

        history.reverse()

        possible_diseases = "\n".join(f"- {hypo}" for hypo in self.hypothesis_space)
        prologue = (
            dedent("""\
            You are an expert medical doctor, and your patient self-reports that: {self_report}. 
               
            The patient is suffering from one of the following possible diseases:      
            {possible_diseases}
    
            You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions. 
            """)
            .format(self_report=self.self_report, possible_diseases=possible_diseases)
            .strip()
        )
        prompt_parts.append(prologue)

        if history:
            conversation_history = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            previous_questions_text = (
                dedent("""\
                These are the questions you've asked to the patient so far:
                {history}
                """)
                .format(history=conversation_history)
                .strip()
            )
            prompt_parts.append(previous_questions_text)

        belief_state = "\n".join(
            f"- Disease: {hypo}; Probability: {prob}"
            for hypo, prob in current_node.belief_state.items()
        )
        generation_prompt = (
            dedent("""\
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases, which are listed along with their probabilities:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            ONLY ASK QUESTIONS WHERE THE ANSWER IS YES OR NO. IF THE ANSWER IS ANY OTHER WORD DO NOT ASK IT.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
            """)
            .format(belief=belief_state, num_questions=self.max_question_nodes)
            .strip()
        )
        prompt_parts.append(generation_prompt)

        return "\n\n".join(prompt_parts)

    @override
    def parse_question_generation_output(self, output: str) -> list[str]:
        question_texts: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )
        return [
            question_text.strip().replace("?", "") + "?"
            for question_text in question_texts
        ]

    @override
    def get_likelihood_elicitation_prompt(self, question: str) -> str:
        hypothesis_space = "\n".join(f"- {hypo}" for hypo in self.hypothesis_space)

        return (
            dedent("""\
            You are an expert medical doctor. 
            You need to estimate the probability of a "Yes" answer for a list of different potential diseases a patient is suffering from, given a single question.

            The question is:
            "{candidate_question}"

            For each of the following diseases, estimate the probability that a patient would answer "Yes" to the question above if they were indeed suffering from that disease.

            Diseases:
            {hypothesis_space}

            Please provide your response ONLY as a single JSON object. The keys should be the disease and the values should be the estimated probability (a float between 0.0 and 1.0).

            Example format for the question "Do you have a fever?" (THE FOLLOWING PROBABILITIES ARE HYPOTHETICAL):
            {{
            "Flu": 0.33,
            "Penumonia": 0.33,
            "Rubella": 0.33,
            "Anemia": 0.0
            }}
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=hypothesis_space,
                candidate_question=question,
            )
            .strip()
        )

    @override
    def parse_likelihood_elicitation_output(
        self, output: str
    ) -> dict[str, dict[str, float]]:
        cleaned_output = output[output.rfind("{") : output.rfind("}") + 1]
        probs = json.loads(cleaned_output)
        return {"Yes": probs, "No": {item: 1 - prob for item, prob in probs.items()}}

    @override
    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        return (
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you questions, and you should answer each one truthfully based on your disease, by saying 'Yes' or 'No'. 
            ONLY ANSWER WITH YES OR NO.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question_node.question)
            .strip()
        )
