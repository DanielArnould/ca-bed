import json
import logging
import re
from textwrap import dedent
from typing import override
from models import Model
from node import EvidenceNode, QuestionNode
from tasks.task import Task

LOGGER = logging.getLogger("Task")


class Bayesian(Task):
    def __init__(
        self,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
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
        return f"Twenty Questions (Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=}"

    @override
    async def create_root(self, model: Model) -> tuple[EvidenceNode, int, int]:
        uniform_probability = 1.0 / len(self.hypothesis_space)
        uniform_prior = {
            hypothesis: uniform_probability for hypothesis in self.hypothesis_space
        }

        return EvidenceNode("ROOT", uniform_prior, 1.0), 0, 0

    @override
    def get_question_generation_prompt(self, current_node: EvidenceNode) -> str:
        history = []
        node = current_node
        while node.parent:
            question = node.parent.question
            answer = node.answer
            history.append((question, answer))
            node = node.parent.parent

        history.reverse()

        bullets = "\n".join(f"- {h}" for h in self.hypothesis_space)
        prologue = (
            dedent("""
                You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X.
                You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            """)
            .format(hypothesis=bullets)
            .strip()
        )

        bullets_history = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
        bullets_belief = "\n".join(
            f"- X: {hypo}; Probability: {prob}"
            for hypo, prob in current_node.belief_state.items()
        )

        if len(bullets_history) == 0:
            generation_prompt = (
                dedent("""
                Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
                {belief}

                Your task is to generate at most {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """)
                .format(
                    history=bullets_history,
                    belief=bullets_belief,
                    num_questions=self.max_question_nodes,
                )
                .strip()
            )
        else:
            generation_prompt = (
                dedent("""
                The game has proceeded as follows:
                {history}

                Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
                {belief}

                Your task is to generate at most {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """)
                .format(
                    history=bullets_history,
                    belief=bullets_belief,
                    num_questions=self.max_question_nodes,
                )
                .strip()
            )

        return f"{prologue}\n\n{generation_prompt}"

    @override
    def parse_question_generation_output(self, output: str) -> list[str]:
        question_texts: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )
        questions = [
            question_text.strip().replace("?", "") + "?"
            for question_text in question_texts
        ]
        assert len(questions) > 0, "No questions generated!"
        return questions

    @override
    def get_likelihood_elicitation_prompt(self, question_node: QuestionNode) -> str:
        bullets = "\n".join(
            f"- {h}" for h in list(question_node.parent.belief_state.keys())
        )

        return (
            dedent("""
            Assume you are playing a game of 20 Questions. You need to estimate the probability of a "Yes" answer for a list of different potential secret objects, given a single question.

            The question is:
            "{candidate_question}"

            For each of the following items, estimate the probability that a person would answer "Yes" to the question above if that item were the secret object.

            Items:
            {hypothesis}

            Please provide your response ONLY as a single JSON object. The keys should be the item names and the values should be the estimated probability (a float between 0.0 and 1.0).

            Example format for the question "Is it an animal?" (THE FOLLOWING PROBABILITIES ARE HYPOTHETICAL):
            {{
            "Dog": 0.99,
            "Cookie": 0.0,
            "Paint": 0.0,
            "Hat": 0.0
            }}
            """)
            .format(hypothesis=bullets, candidate_question=question_node.question)
            .strip()
        )

    @override
    def parse_likelihood_elicitation_output(
        self, output: str
    ) -> dict[str, dict[str, float]]:
        cleaned_output = output[output.find("{") : output.rfind("}") + 1]
        probs: dict = json.loads(cleaned_output)
        confidence_in_val = 1.0
        likelihoods = {
            item: {
                "Yes": (
                    adjusted_prob := (
                        (confidence_in_val * max(min(prob, 1 - (1e-10)), 1e-10))
                        + (1 - confidence_in_val) * 0.5
                    )
                ),
                "No": 1 - adjusted_prob,
            }
            for item, prob in probs.items()
        }
        assert len(likelihoods) > 0, f"Likelihoods empty! {output}"
        return likelihoods

    @override
    def get_answer_selection_prompt(self, question_node: QuestionNode) -> str:
        return (
            dedent("""
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {target_item}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X, by saying 'Yes' or 'No'.
            ONLY ANSWER WITH YES OR NO.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=question_node.question)
            .strip()
        )
