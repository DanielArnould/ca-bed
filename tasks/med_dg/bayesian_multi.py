import json
import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task


class BayesianWithMultibranching(Task):
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
            max_evidence_nodes=2,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Bayesian + Multibranching): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=} {self.self_report=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        prompt = (
            dedent("""\
            You are an expert Doctor, you are given this self-report:
            {self_report}
            
            Below is the set of possible conditions the patient may have:
            {hypothesis_space}
            
            Given your knowledge in this area and the given self-report, please assign a prior belief to each 
            item in the hypothesis set provided above, nothing else. 
            
            For example, if our hypothesis space is [A,B,C,D], and C and A are more relevant given the self-report, you 
            might assign probabilities like below: 
            A: 0.35, B: 0.02, C: 0.55, D: 0.08
            
            Note the probabilities are just conjectures, you should generate reasonable probabilities for each option. 
            Also note that the sum of these prior probabilities is 1. Please give a probability for each hypothesis even 
            if they are extremely small.
            
            Please strictly return your response in the format below:
            {{
                "<Hypothesis 1 full name>": <probability>, 
                "<Hypothesis 2 full name>": <probability>, 
                ....
            }}    
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=self.hypothesis_space,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)
        cleaned_output = output[output.rfind("{") : output.rfind("}") + 1]
        probs: dict = json.loads(cleaned_output)
        adjusted_prior = {
            h: (max(min(p, 1 - (1e-10)), 1e-10)) for h, p in probs.items()
        }
        total = sum(adjusted_prior.values())
        adjusted_prior = {h: v / total for h, v in adjusted_prior.items() if v >= 0.001}

        return adjusted_prior

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts = []

        # Prologue
        parts.append(
            dedent("""\
            You are an expert medical doctor, and your patient self-reports that: {self_report}. 
    
            You should ask your patient questions in English about symptoms in order to find what disease this patient suffers from. 
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
                Your task is to generate {num_questions} *excellent* questions to ask next, along with a list of possible answers.
                The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>|Answer1;Answer2;Answer3
                2. <Question 2>|Answer1;Answer2
                ...
                n. <Question n>|Answer1;Answer2;Answer3;...;AnswerK
                """)
            .format(num_questions=self.max_question_nodes)
            .strip()
        )

        # Query LLM
        prompt = "\n".join(parts)
        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        matches: list[tuple[str, str]] = re.findall(
            r"\d+\.\s*(.*?)\|(.*)", output, re.MULTILINE
        )

        questions = {
            question: [a.strip() for a in answers.split(";")]
            for question, answers in matches
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
            You are an expert medical doctor estimating the likelihood of each possible patient answer to a diagnostic question, for each of several candidate diseases.

            ### Checklist (conceptual)
            - Interpret the diagnostic question and possible answers.
            - Apply relevant medical knowledge for each disease.
            - Estimate probability distribution of answers per disease.
            - Verify probabilities sum to 1.0 (±0.01 tolerance).
            - Ensure outputs strictly match required format.

            ### Inputs
            - Question: "{candidate_question}"
            - Possible answers (in order): {possible_answers}
            - Diseases: {hypothesis_space}

            ### Instructions
            - For each disease, output probabilities over possible answers.
            - Probabilities must:
                - Sum to 1.0 (±0.01 tolerance).
                - Align with the order of {possible_answers}.
                - Be rounded to two decimals (e.g., 0.75).
            - Verify:
                - Number of probabilities = number of possible answers.
                - Probabilities sum to 1.0 within tolerance.
                - If verification fails, output only one of:
                - `"ERROR: Number of probabilities does not match the number of possible answers."`
                - `"ERROR: Probabilities do not sum to 1.0 for [Disease Name]."`

            ### Response Format
            One line per disease:
            <sequence_number>. <Disease Name>|<prob_answer_1>;<prob_answer_2>;...;<prob_answer_n>
                
            **Example:**

            1. Influenza|0.80;0.10;0.10
            2. Common Cold|0.50;0.40;0.10

            Do not include commentary or explanations—return only the formatted response.

            ### Output Schema
            - Input:  
                - candidate_question: str (e.g., "Do you have a cough?")  
                - possible_answers: list[str] (e.g., ["Yes", "No", "Sometimes"])  
                - hypothesis_space: list[str] (e.g., ["Influenza", "Common Cold"])  

            - Output:  
                - For each disease: `<Disease Name>|<probabilities>`  
                - If format check fails: output only the error message.
            """)
            .format(
                possible_answers=answers,
                hypothesis_space=hypotheses,
                candidate_question=question,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        matches: list[tuple[str, str]] = re.findall(
            r"\d+\. ([^|]+)\|([\d.;]+)", output, re.MULTILINE
        )

        raw_likelihoods = {
            disease: {
                ans.strip(): max(min(float(p.strip()), 1 - (1e-10)), 1e-10)
                for ans, p in zip(answers, probs.split(";"))
            }
            for disease, probs in matches
        }

        likelihoods = {
            disease: {
                ans: likelihood / sum(per_disease_likelihoods.values())
                for ans, likelihood in per_disease_likelihoods.items()
            }
            for disease, per_disease_likelihoods in raw_likelihoods.items()
        }

        assert len(likelihoods) > 0, "No likelihoods parsed!"
        return likelihoods

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        # Query LLM
        answers = [child.answer for child in current_node.children]
        prompt = (
            dedent("""\
            You are a patient experiencing {target_item}, and I am your doctor.  
            I will ask you questions about your condition.  

            ### Instructions
            - Answer truthfully based on your symptoms.  
            - Review the available options before responding.  
            - Respond using only one of the provided choices: {answers}  

            ### Question
            {question}
            """)
            .format(
                target_item=self.task_answer,
                answers=answers,
                question=current_node.question,
            )
            .strip()
        )

        output = await query_llm(prompt, self.answerer_session)

        # Parse LLM
        llm_answer = output.strip().lower()
        for child in current_node.children:
            if child.answer.strip().lower() == llm_answer:
                return child

        raise RuntimeError(
            f"No matching answer selected for '{current_node.question}'. Possible answers: {answers}, Given answer: {llm_answer}"
        )
