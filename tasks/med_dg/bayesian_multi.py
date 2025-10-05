from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.med_dg.data import MED_DG_SET, MedDGInstance
from tasks.task import (
    Task,
    parse_answer,
    parse_likelihoods,
    parse_probabilities,
    parse_questions,
)


class BayesianWithMultibranching(Task):
    instance: MedDGInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: MedDGInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
    ):
        self.instance = instance
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=instance.disease,
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=MED_DG_SET,
        )

    def __str__(self) -> str:
        return f"MedDG (Bayesian + Multibranching): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=} {self.instance.self_report=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        prompt = dedent(f"""\
            You are an expert doctor. You are given the following patient self-report:
            {self.instance.self_report}

            ### Task
            - Assign a probability to each condition based on the self-report and your medical knowledge.
            - Every condition in {self.hypothesis_space} must receive a probability, even if very small.
            - Probabilities must sum to 1.0 (±0.01 tolerance).
            - Express each probability as a decimal rounded to two places (e.g., 0.35).
            - Return only the formatted response; no explanations or commentary.

            ### Response Format
            One line per condition:
            <number>. <Condition Name>|<probability>

            ### Example
            1. Influenza|0.35  
            2. Common Cold|0.25  
            3. Pneumonia|0.30  
            4. Allergies|0.10   
            """).strip()

        output = await query_llm(prompt, self.questioner_session)
        priors = parse_probabilities(output)

        return {prior.hypothesis: prior.probability for prior in priors}

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts = []

        # Prologue
        parts.append(
            dedent(f"""\
            You are an expert medical doctor, and your patient self-reports that: {self.instance.self_report}. 
    
            You should ask your patient questions in English about symptoms in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions. 
            """).strip()
        )

        # Conversation History
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent(f"""
                These are the questions you've asked to the patient so far:
                {history_formatted}
                """).strip()
            )

        # Current belief state
        belief_state_formatted = "\n".join(
            f"- Disease: {hypo}; Probability: {prob}"
            for hypo, prob in current_node.belief_state.items()
        )
        parts.append(
            dedent(f"""\
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases, which are listed along with their probabilities:
            {belief_state_formatted}
            """).strip()
        )

        # Question generation
        parts.append(
            dedent(f"""
                Your task is to generate {self.max_question_nodes} *excellent* questions to ask next, along with a list of possible answers.
                The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>|Answer1;Answer2;Answer3
                2. <Question 2>|Answer1;Answer2
                ...
                n. <Question n>|Answer1;Answer2;Answer3;...;AnswerK
                """).strip()
        )

        # Query LLM
        prompt = "\n\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)
        questions = parse_questions(output)

        return {question.question: question.possible_answers for question in questions}

    @override
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        # Query LLM
        prompt = dedent(f"""\
            You are an expert medical doctor diagnosing a patient.
                        
            ### Possible diseases
            {hypotheses}
                        
            ### Question
            "{question}"

            ### {len(answers)} Possible Answers
            {answers}

            ### Task
            For each disease, estimate the likelihood that the patient would give each answer,
            assuming they had that disease. Probabilities must:
            - Align with the provided answers in order
            - Sum to 1.0 (±0.01)
            - Be rounded to two decimals

            If verification fails, return only one of:
            - "ERROR: Number of probabilities does not match the number of possible answers."
            - "ERROR: Probabilities do not sum to 1.0."

            ### Response Format
            One line per disease:
            <sequence_number>. <Disease Name>|<prob_answer_1>;<prob_answer_2>;...;<prob_answer_n>
                
            ### Example

            1. Influenza|0.80;0.10;0.10
            2. Common Cold|0.50;0.40;0.10

            Do not include commentary or explanations—return only the formatted response.
            """).strip()

        # Query LLM
        output = await query_llm(prompt, self.questioner_session)
        likelihoods = parse_likelihoods(output)
        return {
            likelihood.hypothesis: {
                ans: prob
                for ans, prob in zip(answers, likelihood.likelihoods, strict=True)
            }
            for likelihood in likelihoods
        }

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        answers = [child.answer for child in current_node.children]
        prompt = dedent(f"""\
            You are a patient experiencing {self.instance.disease}. You self-reported that: {self.instance.self_report}.
            I am your doctor and I will ask you questions about your condition.  

            ### Instructions
            - Answer truthfully based on your symptoms.  
            - Review the available options before responding.  
            - You must ONLY respond with one of the following option, matching it EXACTLY: {answers}
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Question
            "{current_node.question}"
            """).strip()

        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)
