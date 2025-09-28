import json
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
            max_evidence_nodes=2,
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
    async def create_questions(self, current_node: EvidenceNode) -> list[str]:
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
            dedent("""\
            You are an expert medical doctor. 
            You need to estimate the probability of a "Yes" answer for a list of different potential diseases a patient is suffering from, given a single question.

            The question is:
            "{candidate_question}"

            For each of the following diseases, estimate the probability that a patient would answer "Yes" to the question above if they were indeed suffering from that disease.

            Diseases:
            {hypothesis_space}

            Please provide your response ONLY as a single JSON object. The keys should be the disease and the values should be the estimated probability (a float between 0.0 and 1.0).

            Example format for the question "Do you have a fever?" (THE FOLLOWING PROBABILITIES ARE HYPOTHETICAL):
            {{
            "Flu": 0.33,
            "Penumonia": 0.33,
            "Rubella": 0.33,
            "Anemia": 0.0
            }}
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=hypotheses_formatted,
                candidate_question=question,
            )
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
            dedent("""\
            You are the patient suffering from {target_item}, and I am the doctor. 
            I will ask you questions, and you should answer each one truthfully based on your disease, by saying 'Yes' or 'No'. 
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
