import json
import re
from textwrap import dedent
from typing import override

from node import EvidenceNode, QuestionNode
from tasks.task import Task


class Baseline(Task):
    self_report: str

    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
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
            confidence_threshold=1.0,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Non-Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.hypothesis_space=} {self.self_report=}"

    @override
    def get_prior_prompt(self) -> str:
        return (
            dedent("""\
            You are an expert Doctor, you are given this self-report:
            {self_report}
            
            Below is the set of possible conditions the patient may have:
            {hypothesis_space}
            
            Given your knowledge in this area and the given self-report, please pick from the conditions above those
            which might be relevant, nothing else. You must choose at least 1.
            
            Please strictly return your response in the format below:
            {{
                "conditions": ["Disease A", "Disease B", ..., "Disease N"]
            }}    
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=self.hypothesis_space,
            )
            .strip()
        )

    @override
    def parse_prior_output(self, output: str) -> dict[str, float]:
        cleaned_output = output[output.rfind("{") : output.rfind("}") + 1]
        data = json.loads(cleaned_output)
        relevant_conditions = data["conditions"]
        adjusted_prior = {
            h: 1.0 if h in relevant_conditions else 0 for h in self.hypothesis_space
        }
        total = sum(adjusted_prior.values())
        adjusted_prior = {h: v / total for h, v in adjusted_prior.items()}
        return adjusted_prior

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

        pruned_hypothesis_space = [
            item for item, prob in current_node.belief_state.items() if prob > 1e-5
        ]
        bullets_pruned_hypothesis = "\n".join(
            f"- {item}" for item in pruned_hypothesis_space
        )
        generation_prompt = (
            dedent("""\
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            ONLY ASK QUESTIONS WHERE THE ANSWER IS YES OR NO. IF THE ANSWER IS ANY OTHER WORD DO NOT ASK IT.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
            """)
            .format(
                belief=bullets_pruned_hypothesis, num_questions=self.max_question_nodes
            )
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

            Here are all the possible diseases that the patient may suffer from:
            {items_str}

            Classify the disease based on this single yes/no question:
            Question: "{question}"

            If the answer would be YES when the patient is indeed suffering from that disease, put it in YES; otherwise put it in NO.
            Use the disease names exactly as listed. Cover all diseases exactly once (no omissions, no duplicates).

            Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

            Question 1: <question>
            YES: aaaa, bbbb, ...
            Count of YES: <integer>
            NO: cccc, dddd, ...
            Count of NO: <integer>
            """)
            .format(
                items_str=hypothesis_space,
                question=question,
            )
            .strip()
        )

    @override
    def parse_likelihood_elicitation_output(
        self, output: str
    ) -> dict[str, dict[str, float]]:
        # grab text after YES: until "Count"
        m_yes = re.search(r"YES:\s*(.*?)\s*Count", output, re.DOTALL)
        m_no = re.search(r"NO:\s*(.*?)\s*Count", output, re.DOTALL)

        yes_set = set([s.strip() for s in m_yes.group(1).split(",")] if m_yes else [])
        no_set = set([s.strip() for s in m_no.group(1).split(",")] if m_no else [])
        hs = set(self.hypothesis_space)
        likelihoods_yes = {h: (1.0 if h in yes_set else 0.0) for h in hs}
        likelihoods_no = {h: (1.0 if h in no_set else 0.0) for h in hs}

        return {"Yes": likelihoods_yes, "No": likelihoods_no}

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
