import re
from textwrap import dedent
from typing import override
from node import EvidenceNode, QuestionNode
from tasks.task import Task


class Baseline(Task):
    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        hypothesis_space: list[str],
    ):
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
        return f"Twenty Questions (Non-Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.hypothesis_space=}"

    @override
    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append((question, answer))
            node = node.parent.parent

        prologue = dedent("""
            You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X.
            You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            Let us begin. Ask me the first question.
            """).strip()

        bullets_history = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
        pruned_hypothesis = [
            item for item, prob in current_node.belief_state.items() if prob > 0
        ]
        bullets_pruned_hypothesis = "\n".join(f"- {item}" for item in pruned_hypothesis)

        if len(bullets_history) == 0:
            generation_prompt = (
                dedent("""
                Based on our current beliefs, the secret object is most likely one of the following items:
                {belief}

                Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """)
                .format(
                    history=bullets_history,
                    belief=bullets_pruned_hypothesis,
                    num_questions=self.max_question_nodes,
                )
                .strip()
            )

        generation_prompt = (
            dedent("""
            The game has proceeded as follows:
            {history}

            Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
            """)
            .format(
                history=bullets_history,
                belief=bullets_pruned_hypothesis,
                num_questions=self.max_question_nodes,
            )
            .strip()
        )

        # TODO: Investigate targeting prompts
        # target_prompt = ""
        # if get_conversation_depth(current_node) >= 14:
        #     target_prompt = baseline_prompts.get_targeting_prompt()

        return f"{prologue}\n\n{generation_prompt}"

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
        items_str = "\n".join(f"- {x}" for x in self.hypothesis_space)

        return (
            dedent("""
            Here are all the X:
            {items_str}

            Classify the X based on this single yes/no question:
            Question: "{question}"

            If the answer would be YES when that X is the secret object, put it in YES; otherwise put it in NO.
            Use the item strings exactly as listed. Cover all items exactly once (no omissions, no duplicates).

            Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

            Question 1: <question>
            YES: aaaa, bbbb, ...
            Count of YES: <integer>
            NO: cccc, dddd, ...
            Count of NO: <integer>
        """)
            .format(items_str=items_str, question=question)
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
            dedent("""
            You are an expert player of the 20 Questions game. Your goal is to impersonate the secret object, X. I will be trying to guess the secret object, X. X is {target_item}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X, by saying 'Yes' or 'No'.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question_node.question)
            .strip()
        )

    @override
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
