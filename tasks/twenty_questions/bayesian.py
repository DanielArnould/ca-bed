import json
import re
from textwrap import dedent
from typing import override
from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task


class Bayesian(Task):
    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_question_nodes=max_question_nodes,
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"Twenty Questions (Bayesian): {self.questioner_session.model_key=} {self.answerer_session.model_key=} {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        uniform_probability = 1.0 / len(self.hypothesis_space)
        return {hypothesis: uniform_probability for hypothesis in self.hypothesis_space}

    @override
    async def create_questions(self, current_node: EvidenceNode) -> list[str]:
        parts = []

        # Prologue
        parts.append(
            dedent("""
                You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X.
                You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            """).strip()
        )

        # Conversation History
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent("""
                The game has proceeded as follows:
                {history}
                """)
                .format(history=history_formatted)
                .strip()
            )

        # Current belief state
        belief_state_formatted = "\n".join(
            f"- X: {hypo}; Probability: {prob}"
            for hypo, prob in current_node.belief_state.items()
        )
        parts.append(
            dedent("""
                Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
                {belief}
                """)
            .format(belief=belief_state_formatted)
            .strip()
        )

        # Question generation
        parts.append(
            dedent("""
                Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """)
            .format(num_questions=self.max_question_nodes)
            .strip()
        )

        # Query LLM
        prompt = "\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        question_texts: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )
        questions = [question_text.strip() for question_text in question_texts]
        assert len(questions) > 0, "No questions generated!"
        return questions

    @override
    async def get_likelihoods(
        self, current_node: QuestionNode
    ) -> dict[str, dict[str, float]]:
        # Query LLM
        hypotheses = "\n".join(
            f"- {h}" for h in current_node.parent.belief_state.keys()
        )

        prompt = (
            dedent("""
            Assume you are playing a game of 20 Questions. You need to estimate the probability of a "Yes" answer for a list of different potential secret objects, given a single question.

            The question is:
            "{candidate_question}"

            For each of the following items, estimate the probability that a person would answer "Yes" to the question above if that item were the secret object.

            Items:
            {hypotheses}

            Please provide your response ONLY as a single JSON object. The keys should be the item names and the values should be the estimated probability (a float between 0.0 and 1.0).

            Example format for the question "Is it an animal?" (THE FOLLOWING PROBABILITIES ARE HYPOTHETICAL):
            {{
            "Dog": 0.99,
            "Cookie": 0.0,
            "Paint": 0.0,
            "Hat": 0.0
            }}
            """)
            .format(candidate_question=current_node.question, hypotheses=hypotheses)
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        cleaned_output = output[output.find("{") : output.rfind("}") + 1]
        probs: dict = json.loads(cleaned_output)
        likelihoods = {
            item: {
                "Yes": (adjusted_prob := (max(min(prob, 1 - (1e-10)), 1e-10))),
                "No": 1 - adjusted_prob,
            }
            for item, prob in probs.items()
        }
        assert len(likelihoods) > 0, f"Likelihoods empty! {output}"
        return likelihoods

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        # Query LLM
        prompt = (
            dedent("""
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {target_item}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X, by saying 'Yes' or 'No'.
            ONLY ANSWER WITH YES OR NO.
            Let us begin. Here is my question:
            {question}
            """)
            .format(target_item=self.task_answer, question=current_node.question)
            .strip()
        )

        output = await query_llm(prompt, self.answerer_session)

        # Parse LLM
        llm_answer = output.strip().lower()
        for child in current_node.children:
            if child.answer.lower() in llm_answer:
                return child

        raise RuntimeError(
            f"No matching answer selected for '{current_node.question}'. Possible answers: {list(child.answer for child in current_node.children)}, Given answer: {llm_answer}"
        )
