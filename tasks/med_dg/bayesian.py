import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task


class Bayesian(Task):
    self_report: str

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
        self_report: str,
    ):
        self.self_report = self_report
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
        return f"MedDG (Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=} {self.self_report=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        prompt = (
            dedent("""\
            You are an expert doctor. You are given the following patient self-report:
            {self_report}

            ### Task
            - Assign a prior probability to each condition based on the self-report and your medical knowledge.
            - Every condition in {hypothesis_space} must receive a probability, even if very small.
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
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=self.hypothesis_space,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        matches: list[tuple[str, str]] = re.findall(
            r"\d+\. ([^|]+)\|([\d.]+)", output, re.MULTILINE
        )

        raw_priors = {
            disease: max(min(float(prob.strip()), 1 - (1e-10)), 1e-10)
            for disease, prob in matches
        }

        priors = {
            disease: prob / sum(raw_priors.values())
            for disease, prob in raw_priors.items()
        }

        assert len(priors) > 0, "No priors parsed!"
        return priors

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts = []

        # Prologue
        parts.append(
            dedent("""\
            You are an expert medical doctor, and your patient self-reports that: {self_report}. 
    
            You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
            Use the ongoing conversation for context to avoid redundant questions. 
            """)
            .format(self_report=self.self_report)
            .strip()
        )

        # Conversation History
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent("""
                These are the questions you've asked to the patient so far:
                {history}
                """)
                .format(history=history_formatted)
                .strip()
            )

        # Current belief state
        belief_state_formatted = "\n".join(
            f"- Disease: {hypo}; Probability: {prob}"
            for hypo, prob in current_node.belief_state.items()
        )
        parts.append(
            dedent("""\
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases, which are listed along with their probabilities:
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
        questions = {
            question_text.strip(): ["Yes", "No"] for question_text in question_texts
        }
        assert len(questions) > 0, "No questions generated!"
        return questions

    @override
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        # Query LLM
        prompt = (
            dedent("""\
            You are an expert medical doctor estimating the likelihood of a "yes" answer to a diagnostic question for each candidate disease.

            ### Checklist (conceptual)
            - Interpret the diagnostic question.  
            - Apply relevant medical knowledge for each disease.  
            - Estimate the probability a patient would answer "yes" if they have the disease.  
            - Ensure outputs strictly match the required format.

            ### Inputs
            - Question: "{candidate_question}"  
            - Diseases: {hypothesis_space}

            ### Instructions
            - For each disease, output the probability of a "yes" response.  
            - Probabilities must be rounded to two decimals (e.g., 0.75).  

            ### Response Format
            One line per disease:
            <sequence_number>. <Disease Name>|<probability>

            **Example:**
            1. Influenza|0.80  
            2. Common Cold|0.50  

            Return only the formatted response; do not include commentary or explanations.

            ### Output Schema
            - Input:  
                - candidate_question: str (e.g., "Do you have a cough?")  
                - hypothesis_space: list[str] (e.g., ["Influenza", "Common Cold"])  

            - Output:  
                - For each disease: `<Disease Name>|<probability>`
            """)
            .format(
                hypothesis_space=hypotheses,
                candidate_question=question,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        matches: list[tuple[str, str]] = re.findall(
            r"\d+\. ([^|]+)\|([\d.]+)", output, re.MULTILINE
        )

        likelihoods = {
            disease: {
                "Yes": (
                    adjusted_prob := max(min(float(prob.strip()), 1 - (1e-10)), 1e-10)
                ),
                "No": 1 - adjusted_prob,
            }
            for disease, prob in matches
        }

        assert len(likelihoods) > 0, "No likelihoods parsed!"
        return likelihoods

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        # Query LLM
        prompt = (
            dedent("""\
            You are a patient experiencing {target_item}, and I am your doctor.  
            I will ask you questions about your condition.  

            ### Instructions
            - Answer truthfully based on your symptoms.  
            - Review the available options before responding.  
            - Respond using only 'Yes' or 'No'. 

            ### Question
            {question}
            """)
            .format(target_item=self.task_answer, question=current_node.question)
            .strip()
        )

        output = await query_llm(prompt, self.answerer_session)

        # Parse LLM
        llm_answer = output.strip().lower()
        for child in current_node.children:
            if child.answer.strip().lower() == llm_answer:
                return child

        raise RuntimeError(
            f"No matching answer selected for '{current_node.question}'. Possible answers: {list(child.answer for child in current_node.children)}, Given answer: {llm_answer}"
        )
