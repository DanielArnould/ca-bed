import re
from textwrap import dedent
from typing import override
from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task


class Baseline(Task):
    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        hypothesis_space: list[str],
    ):
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=1.0,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"Twenty Questions (Non-Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.hypothesis_space=}"

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
            f"- {item}" for item in current_node.belief_state.keys()
        )
        parts.append(
            dedent("""
                Based on our current beliefs, the secret object is most likely one of the following items:
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
        self, question: str, hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        # Query LLM
        hypotheses_formatted = "\n".join(f"- {h}" for h in hypotheses)

        prompt = (
            dedent("""
            Here are all the X:
            {hypotheses}

            Classify the X based on this single yes/no question:
            Question: "{candidate_question}"

            If the answer would be YES when that X is the secret object, put it in YES; otherwise put it in NO.
            Use the item strings exactly as listed. Cover all items exactly once (no omissions, no duplicates).

            Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

            Question 1: <question>
            YES: aaaa, bbbb, ...
            Count of YES: <integer>
            NO: cccc, dddd, ...
            Count of NO: <integer>
            """)
            .format(
                candidate_question=question,
                hypotheses=hypotheses_formatted,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        # grab text after YES: until "Count"
        m_yes = re.search(r"YES:\s*(.*?)\s*Count", output, re.DOTALL)
        m_no = re.search(r"NO:\s*(.*?)\s*Count", output, re.DOTALL)

        yes_set = set([s.strip() for s in m_yes.group(1).split(",")] if m_yes else [])
        no_set = set([s.strip() for s in m_no.group(1).split(",")] if m_no else [])
        hs = yes_set.union(no_set)

        return {
            h: {"Yes": 1 - (1e-10), "No": 1e-10}
            if h in yes_set
            else {"Yes": 1e-10, "No": 1 - (1e-10)}
            for h in hs
        }

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
