import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.task import Task


class BaselineWithMultibranching(Task):
    self_report: str

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
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
            confidence_threshold=1.0,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"MedDG (Non-Bayesian + Multibranching): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.hypothesis_space=} {self.self_report=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        prompt = (
            dedent("""\
            You are an expert doctor. You are given the following patient self-report:
            {self_report}

            ### Task
            - Select one or more conditions from {hypothesis_space} that the patient may have.  
            - You must select at least one.  
            - Return only the formatted response; no explanations or commentary.  

            ### Response Format
            One line per selected condition:
            <number>. <Condition Name>

            ### Example
            1. Influenza  
            2. Common Cold  
            3. Pneumonia  
            4. Allergies 
            """)
            .format(
                self_report=self.self_report,
                hypothesis_space=self.hypothesis_space,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        matches: list[str] = re.findall(
            r"\d+\.\s+(.*?)(?=\s*\d+\.|$)", output, re.MULTILINE
        )

        selected_conditions = [disease.strip() for disease in matches]
        priors = {
            disease: 1 / len(selected_conditions) for disease in selected_conditions
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
            f"- {hypo}" for hypo in current_node.belief_state.keys()
        )
        parts.append(
            dedent("""\
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases:
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
            You are an expert medical doctor classifying diseases based on a diagnostic question.

            ### Checklist (conceptual)
            - Interpret the diagnostic question and its possible answers.  
            - Apply medical knowledge for each disease.  
            - Assign each disease to the answer a patient with that disease would most likely give.  
            - Ensure every disease is classified exactly once (no omissions, no duplicates).  
            - Follow the required output format exactly.  

            ### Inputs
            - Question: "{candidate_question}"  
            - Possible answers (in order): {possible_answers}  
            - Diseases: {hypothesis_space}  

            ### Instructions
            - Place each disease under one of the provided answers.  
            - Use the disease names exactly as given.  

            ### Response Format
            
            <ANSWER1>: disease_1, disease_2, ...
            Count of <ANSWER1>: <integer>
            <ANSWER2>: disease_3, disease_4, ...
            Count of <ANSWER2>: <integer>
                   
            **Example:**
            
            YES: Influenza, Common Cold, Pneumonia
            Count of YES: 3
            NO: Allergies
            Count of NO: 1

            Return only the formatted response; do not include commentary or explanations.

            ### Output Schema
            - Input:  
                - candidate_question: str (e.g., "Do you have a cough?")  
                - possible_answers: list[str] (e.g., ["Yes", "No", "Sometimes"])  
                - hypothesis_space: list[str] (e.g., ["Influenza", "Common Cold", "Pneumonia", "Allergies"])  

            - Output:  
                - Each possible answer label followed by its assigned diseases.  
                - Each answer section must include a count of diseases.  
            """)
            .format(
                hypothesis_space=hypotheses,
                possible_answers=answers,
                candidate_question=question,
            )
            .strip()
        )

        output = await query_llm(prompt, self.questioner_session)

        # Parse LLM
        answer_to_diseases = {}

        for answer in answers:
            # Match pattern: "ANSWER: diseases... Count"
            pattern = rf"{re.escape(answer)}:\s*(.*?)\s*Count"
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

            if match:
                diseases_str = match.group(1).strip()
                diseases = [d.strip() for d in diseases_str.split(",") if d.strip()]
                answer_to_diseases[answer] = set(diseases)
            else:
                answer_to_diseases[answer] = set()

        all_found_diseases = set()
        for diseases in answer_to_diseases.values():
            all_found_diseases.update(diseases)

        raw_likelihoods: dict[str, dict[str, float]] = {}
        for disease in all_found_diseases:
            raw_likelihoods[disease] = {}
            for answer in answers:
                if disease in answer_to_diseases[answer]:
                    raw_likelihoods[disease][answer] = 1 - (1e-10)
                else:
                    raw_likelihoods[disease][answer] = 1e-10

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
