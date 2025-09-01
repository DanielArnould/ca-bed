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
        return f"Flodial (Non-Bayesian): Answer: {self.task_answer}, Self Report: {self.self_report}, Max Question Nodes: {self.max_question_nodes}, Hypothesis Space: {self.hypothesis_space}"

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
            You are a technician, and your client self-reports that: {self_report}.
            You should ask your client question with specific situation which can only be answered by 'Yes' or 'No', in order to find which issue this client faces with.
            Let us begin. Ask me the first question.
            """)
            .format(self_report=self.self_report)
            .strip()
        )
        prompt_parts.append(prologue)

        if history:
            asked = "\n".join(f"- Q: {q}" for q, _ in history)
            previous_questions_text = (
                dedent("""\
                These are the questions you've asked the client so far:
                {asked}
                """)
                .format(asked=asked)
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
            Based on our current beliefs, the issues of the client's issue is one of the following:
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

        if get_conversation_depth(current_node) >= 0.6 * 20:
            target_prompt = dedent("""\
                Note that you should point out and ask what the issue the client's facing for the given report problem is from now.
                The question must be 'Are you experiencing [issue name]?
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
            You are an expert technician, and your client self-reports that: {self_report} 
            
            Here are all issues that the client may face with: {items_str}. 

            Classify the disease based on this single yes/no question:
            Question: "{question}"

            The answer would be YES when the patient is indeed facing that issue, put it in YES; otherwise put it in NO.
            Use the issues names exactly as listed. Cover all issues exactly once (no omissions, no duplicates).

            Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

            Question 1: <question> 
            YES: aaaa, bbbb, ... (issues names only)
            Count of YES: <integer>
            NO: cccc, dddd, ...
            Count of NO: <integer> (issues names only)
            """)
            .format(
                issue=self.self_report, items_str=items_str, question=question
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
            You are the client with a device that has '{target_item}' and I am the technician.
            I will ask you up to 20 questions and you should answer each one truthfully based on the issue of your device.
            If I point out correctly what your issue is, answer me "You are right. My device has '{target_item}'."
            Note that never directly tell me what the issue is all the time.
            Let us begin. Here is my first question.
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
