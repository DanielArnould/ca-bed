import re
from textwrap import dedent

from method import get_conversation_depth
from node import EvidenceNode, QuestionNode
from tasks.craft_md.data import CraftMDInstance
from ..task import Question, Task


class Baseline(Task):
    patient_info: str
    atomic_facts: list[str]

    def __init__(
        self,
        craft_md_instance: CraftMDInstance,
        hypothesis_space: list[str],
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
    ):
        self.patient_info = craft_md_instance.patient_info
        self.atomic_facts = craft_md_instance.atomic_facts
        super().__init__(
            task_answer=craft_md_instance.ground_truth,
            max_question_nodes=max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=1.0,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            f"CraftMD (Non-Bayesian): Answer: {self.task_answer}, Context: {self.patient_info}, "
            f"Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"
        )

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

        prologue = (
            dedent("""\
            You are an expert medical doctor. A patient comes into the clinic presenting with some basic information:
            "{patient_info}" 
    
            You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions.
            """)
            .format(patient_info=self.patient_info)
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
            item for item, prob in current_node.belief_state.items() if prob > 0
        ]
        bullets_pruned_hypothesis = "\n".join(
            f"- {item}" for item in pruned_hypothesis_space
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
            .format(
                belief=bullets_pruned_hypothesis, num_questions=self.max_question_nodes
            )
            .strip()
        )
        prompt_parts.append(generation_prompt)

        if get_conversation_depth(current_node) >= 3:
            target_prompt = dedent("""\
                You *may* now attempt to directly identify the disease if appropriate.
                Refer to the past conversation regarding the patient's symptoms. Never repeat previously asked questions.
                If you think you can identify it, ask a confirmation question starting with 'Are you suffering from ...'
                """).strip()
            prompt_parts.append(target_prompt)

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
        pruned_hypothesis = [
            item for item, prob in current_node.belief_state.items() if prob > 0
        ]
        items_str = "\n".join(f"- {x}" for x in pruned_hypothesis)

        return (
            dedent("""\
            You are an expert diagnostician performing a differential diagnosis.

            ## Patient Information
            "{patient_info}"

            ## Candidate Question
            "{question}"

            ## Current Differential (Diseases Still Under Consideration)
            {items_str}

            ## Your Task
            For EACH disease, decide whether a truthful patient actually suffering from that disease would answer the candidate yes/no question with "Yes" (place it in YES) or "No" (place it in NO).

            Interpret "Yes" as: The symptom / condition / situation implied by the question is expected or commonly present for a patient with that disease.
            Interpret "No" as: The symptom / condition / situation is NOT typically present (or would usually be absent) for a patient with that disease.

            ## Requirements
            - Use every disease name exactly once.
            - Do not omit or invent diseases.
            - No duplicates between the YES and NO lists.
            - Keep disease names exactly as written (case & spacing preserved).
            - Provide counts consistent with the list lengths.

            ## Output Format (STRICT — no extra commentary)
            Question 1: <question>
            YES: disease_a, disease_b, ...
            Count of YES: <integer>
            NO: disease_c, disease_d, ...
            Count of NO: <integer>
            """)
            .format(
                patient_info=self.patient_info, items_str=items_str, question=question
            )
            .strip()
        )

    def parse_likelihood_elicitation_output(
        self, output: str, question
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

    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        atomic_facts = "\n".join(self.atomic_facts)

        return (
            dedent("""\
            You are a truthful patient suffering from: {target_item}. A medical doctor is asking yes/no questions to diagnose you.

            Relevant factual statements about your current condition:
            {atomic_facts}

            Doctor's question: {question}

            IMPORTANT:
                - Do NOT reveal the disease name explicitly unless the question is a direct confirmation (e.g., 'Are you suffering from <disease>?').
                - If the doctor correctly names the disease, reply exactly: "You are right. I am experiencing {target_item}." (and nothing else)
                - Otherwise reply with only 'Yes' or 'No'.
            """)
            .format(
                    target_item=self.task_answer,
                    question=question_node.question,
                    atomic_facts=atomic_facts,
            )
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
