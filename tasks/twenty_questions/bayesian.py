from textwrap import dedent
from typing import override

from tenacity import retry, stop_after_attempt
from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task, parse_answer, parse_binary_questions, parse_probabilities


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
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts = []

        # Prologue
        parts.append(
            dedent("""
                You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X. I will be impersonating the secret entity, X.
                You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            """).strip()
        )

        # Conversation History
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent(f"""
                The game has proceeded as follows:
                {history_formatted}
                """).strip()
            )

        # Current belief state
        belief_state_formatted = "\n".join(
            f"- {item}" for item in current_node.belief_state.keys()
        )
        parts.append(
            dedent(f"""
                The secret object is most likely one of the following items:
                {belief_state_formatted}
                """).strip()
        )


        # Question generation
        parts.append(
            dedent(f"""
                Your task is to generate {self.max_question_nodes} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """).strip()
        )

        # Query LLM
        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)
        questions = parse_binary_questions(output)

        return {question.question: question.possible_answers for question in questions}

    @override
    @retry(stop=stop_after_attempt(2))
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        # Query LLM
        hypotheses_formatted = "\n".join(f"- {h}" for h in hypotheses)
        prompt = dedent(f"""
            You are playing a game of 20 Questions.

            ### Possible entities
            {hypotheses_formatted}

            ### Question
            "{question}"

            ### Task
            For each entity, estimate the likelihood that the answer to the question would be 'Yes',
            assuming that was the entity. Probabilities must:
            - Be rounded to two decimals

            ### Response Format
            One line per entity:
            <sequence_number>. <Entity>|<probability>
                
            ### Example
            1. Dog|0.80
            2. Cookie|0.50

            Do not include commentary or explanations—return only the formatted response.
            """).strip()

        output = await query_llm(prompt, self.questioner_session)

        # Query LLM
        output = await query_llm(prompt, self.questioner_session)
        probabilities = parse_probabilities(output)

        likelihoods = {
            probability.hypothesis: {
                "Yes": probability.probability,
                "No": 1 - probability.probability,
            }
            for probability in probabilities
        }

        return likelihoods

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        prompt = dedent(f"""\
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {self.task_answer}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X.

            ### Instructions
            - Answer truthfully based on what X is.  
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Question
            "{current_node.question}"
            """).strip()

        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)
