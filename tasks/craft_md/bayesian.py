import json
import re
from textwrap import dedent

from node import EvidenceNode, QuestionNode
from tasks.craft_md.data import CraftMDInstance
from ..task import Question, Task


class Bayesian(Task):
    patient_info: str
    atomic_facts: list[str]

    def __init__(
        self,
        craft_md_instance: CraftMDInstance,
        hypothesis_space: list[str],
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float
    ):
        self.patient_info = craft_md_instance.patient_info
        self.atomic_facts = craft_md_instance.atomic_facts
        super().__init__(
            task_answer=craft_md_instance.ground_truth,
            max_question_nodes=max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"CraftMD (Bayesian): Answer: {self.task_answer}, Context: {self.patient_info}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

    def get_initial_belief_state(self) -> dict[str, float]:
        prob = 1.0 / len(self.hypothesis_space)
        return {item: prob for item in self.hypothesis_space}

    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        prompt_parts = []
        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append((question, answer))
            node = node.parent.parent

        hypothesis = "\n".join(f"- {hypo}" for hypo in self.hypothesis_space)
        prologue = (
            dedent("""\
            You are an expert medical doctor. A patient comes into the clinic presenting with some basic information:
            "{patient_info}" 
    
            You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions.
            """)
            .format(patient_info=self.patient_info, hypothesis=hypothesis)
            .strip()
        )
        prompt_parts.append(prologue)

        if history:
            conversation_history = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            previous_questions_text = (
                dedent("""
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

    def parse_question_generation_output(self, output: str) -> list[Question]:
        question_texts: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )
        questions = []

        for question_text in question_texts:
            clean_question_text = question_text.strip().replace("?", "") + "?"
            questions.append(
                Question(question=clean_question_text, answers=["Yes", "No"])
            )

        return questions

    def get_likelihood_elicitation_prompt(
        self, current_node: EvidenceNode, question: Question
    ) -> str:
        json_template_dict = {disease: None for disease in self.hypothesis_space}
        json_template_str = json.dumps(json_template_dict, indent=4).strip('{}').strip()

        return (
            dedent("""\
            You are an expert diagnostician performing a differential diagnosis. Your task is to estimate a set of conditional probabilities based on a patient's information and a specific diagnostic question.

            ## Context

            * **Patient Information**:
                "{patient_info}"

            * **Question Under Consideration**:
                "{candidate_question}"

            ## Your Task

            For each potential disease listed below, estimate the conditional probability that a patient would answer "Yes" to the question IF they were suffering from that specific disease. In other words, estimate **P(Answer="Yes" | Disease)**.

            ## Output Instructions

            Provide your response **ONLY** as a single JSON object. The keys must be the **exact** disease names provided in the template below, and the values must be the estimated probability (a float between 0.0 and 1.0).

            **Fill in the `null` values in this template:**
            ```json
            {{
                {json_template_str}
            }}
            """)
            .format(
                patient_info=self.patient_info,
                json_template_str=json_template_str,
                candidate_question=question.question,
            )
            .strip()
        )

    def parse_likelihood_elicitation_output(
        self, output: str, question: Question
    ) -> dict[str, dict[str, float]]:
        cleaned_output = output[output.find("{") : output.rfind("}") + 1]
        probs = json.loads(cleaned_output)
        return {"Yes": probs, "No": {item: 1 - prob for item, prob in probs.items()}}

    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        atomic_facts = "\n".join(self.atomic_facts)

        return (
            dedent("""\
            You are a truthful patient suffering from: {target_item}. A medical doctor is asking yes/no questions to diagnose you.

            Relevant factual statements about your current condition:
            {atomic_facts}

            Doctor's question: {question}
                   
            Note that you must **never directly reveal** the disease you're suffering from. 
            Based on the list of factual statements above, simply answer the question by **only returning** 'Yes' or 'No'
            """)
            .format(target_item=self.task_answer, question=question_node.question, atomic_facts=atomic_facts)
            .strip()
        )

    def parse_answer_selection_output(
        self, output: str, question_node: QuestionNode
    ) -> EvidenceNode:
        llm_answer = output.strip().lower()
        for child in question_node.children:
            if child.answer.lower() in llm_answer:
                return child

        assert False, (
            f"No matching answer selected. Possible answers: {list(child.answer for child in question_node.children)} Actual answer: {llm_answer}"
        )
