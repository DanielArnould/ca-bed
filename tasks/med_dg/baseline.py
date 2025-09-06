import re
from textwrap import dedent

from method import get_conversation_depth
from node import EvidenceNode, QuestionNode
from ..task import Question, Task


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
        return f"MedDG (Non-Bayesian): Answer: {self.task_answer}, Self Report: {self.self_report}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

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
            You are an expert medical doctor, and your patient self-reports that: {self_report}.
            You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions. 
            Let us begin. Ask me the first question.
            """)
            .format(self_report=self.self_report)
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
                Note that you should point out and ask what disease the patient suffers from now.
                Refer to the past conversation regarding the patient's symptoms. Never repeat previously asked questions.
                The question must start with 'Are you suffering from ...'
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
            You are an expert medical doctor, and your patient self-reports that: {self_report}. 

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
                self_report=self.self_report, items_str=items_str, question=question
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
        return (
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you up to 6 questions, and you should answer each one truthfully based on your disease, by saying 'Yes' or 'No'. 
            Note that you must never reveal the disease until I tell it correctly. 
            If I tell the disease correctly in my question, directly respond: "You are right. I am experiencing {target_item}."
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question_node.question)
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
