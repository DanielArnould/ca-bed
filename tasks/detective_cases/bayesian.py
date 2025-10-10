import re
from textwrap import dedent
from typing import override

from tenacity import retry, stop_after_attempt
from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.task import (
    Task,
    parse_answer,
    parse_binary_questions,
    parse_probabilities,
)


class Bayesian(Task):
    instance: DetectiveCasesInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: DetectiveCasesInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
    ):
        self.instance = instance
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=next(
                suspect["name"]
                for suspect in self.instance["suspects"]
                if suspect.get("is_murderer", False)
            ),
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=[suspect["name"] for suspect in self.instance["suspects"]],
        )

    def __str__(self) -> str:
        return f"Detective Cases (Bayesian): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=}"

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        suspects_info_parts = []
        for idx, suspect in enumerate(self.instance["suspects"], start=1):
            suspects_info_parts.append(
                dedent(f"""\
                - Suspect {idx}:
                    - Name: {suspect["name"]}
                    - Introduction: {suspect["introduction"]}
                """).strip()
            )
        suspects_info = "\n".join(suspects_info_parts)

        prompt = dedent(f"""\
            You will take on the role of a detective tasked with finding the real murderer in this case.

            ### Case Background
            {self.get_case_background()}

            The investigation focuses on {len(self.hypothesis_space)} suspects, one of whom is the true murderer:
            {suspects_info}

            ### Task
            - Assign a probability to each suspect.
            - Every suspect in {self.hypothesis_space} must receive a probability, even if very small.
            - Probabilities must sum to 1.0 (±0.01 tolerance).
            - Express each probability as a decimal rounded to two places (e.g., 0.35).
            - Return only the formatted response; no explanations or commentary.

            ### Response Format
            One line per suspect:
            <number>. <Suspect Name>|<probability>

            ### Example
            1. Suspect A|0.35  
            2. Suspect B|0.25  
            3. Suspect C|0.30  
            4. Suspect D|0.05  
            5. Suspect E|0.05  
            """).strip()

        output = await query_llm(prompt, self.questioner_session)
        priors = parse_probabilities(output)

        return {prior.hypothesis: prior.probability for prior in priors}

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        parts = []

        # Case background
        parts.append(
            dedent(f"""\
            You are a detective investigating a murder.  

            ### Case Background
            {self.get_case_background()}
            """).strip()
        )

        # Suspects info
        suspects_info_parts = []
        for idx, suspect in enumerate(self.instance["suspects"], start=1):
            suspects_info_parts.append(
                dedent(f"""\
                - Suspect {idx}:
                    - Name: {suspect["name"]}
                    - Introduction: {suspect["introduction"]}
                """).strip()
            )
        suspects_info = "\n".join(suspects_info_parts)
        parts.append(
            dedent(f"""\
            The investigation focuses on {len(self.hypothesis_space)} suspects:
            {suspects_info}
            """).strip()
        )

        # Conversation history
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent(f"""\
                These are the questions you've already asked so far:
                {history_formatted}
                """).strip()
            )

        # Current belief state
        belief_state_formatted = "\n".join(
            f"- Suspect: {hypo}; Probability: {prob:.2f}"
            for hypo, prob in current_node.belief_state.items()
        )
        parts.append(
            dedent(f"""\
            Based on the current belief state, the likelihood of each suspect being the murderer is:
            {belief_state_formatted}
            """).strip()
        )

        # Question generation instructions
        parts.append(
            dedent(f"""\
            ### Task
            Generate {self.max_question_nodes} excellent yes/no interrogation questions.  
            - Each question must be explicitly directed to a specific suspect.  
            - Format the question as: "[Suspect Name] Question text", with no ; or | in the question text. 
            - Each question can only answered by 'Yes' or 'No'
            - Focus on questions that help distinguish between suspects (motive, alibi, opportunity, access to weapon).

            ### Response Format
            One line per question:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>

            ### Example
            1. [Mr. Jones] Were you outside at 12:00PM? 
            2. [Dr. Otto] Did you have access to the murder weapon?
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
        answerer_name, actual_question = self.parse_question(question)

        suspects = [s for s in self.instance["suspects"] if s["name"] in hypotheses]
        assert len(suspects) > 0

        suspects_info = "\n".join(
            dedent(f"""\
            - Suspect {idx + 1}:
                - Name: {suspect["name"]}
                - Introduction: {suspect["introduction"]}
            """).strip()
            for idx, suspect in enumerate(self.instance["suspects"])
        )

        # Prompt
        prompt = dedent(f"""\
            You are a detective investigating a murder case.

            ### Case Background
            {self.get_case_background()}

            ### Suspects
            {suspects_info}

            ### Question to {answerer_name}
            "{actual_question}"

            ### Task
            For each suspect, estimate the likelihood that {answerer_name} would answer 'Yes',
            assuming each suspect is the murderer. Probabilities must:
            - Be rounded to two decimals

            ### Response Format
            One line per suspect:
            <sequence_number>. <Suspect Name>|<probability>

            ### Example
            1. Ms. Alice|0.80
            2. Dr. Bob|0.40

            Do not include commentary or explanations. Return only the formatted response.
        """)

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
        suspect_name, question = self.parse_question(current_node.question)

        suspect = next(
            (s for s in self.instance["suspects"] if s["name"] == suspect_name),
            None,
        )
        assert suspect is not None, f"Suspect '{suspect_name}' not found in case data"

        prompt = dedent(f"""\
            You are roleplaying as a suspect in a murder investigation.

            ### Suspect
            - Name: {suspect["name"]}
            - Task: {suspect["task"]}
            - Story: {suspect["story"]}

            ### Instructions
            - Answer the detective's question in character as {suspect_name}.
            - Stay consistent with your task and story.
            - You may lie, evade, or tell the truth depending on what seems natural for this suspect.
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Detective's Question
            "{question}"
        """)

        output = await query_llm(prompt, self.answerer_session)
        return parse_answer(output, current_node)

    def get_case_background(self) -> str:
        return dedent(f"""\
            Time: {self.instance["time"]}
            Location: {self.instance["location"]}
            Victim:
            - Name: {self.instance["victim"]["name"]}
            - Introduction: {self.instance["victim"]["introduction"]}
            - Cause of Death: {self.instance["victim"]["cause_of_death"]}
            - Murder Weapon: {self.instance["victim"]["murder_weapon"]}
        """)

    def parse_question(self, question: str) -> tuple[str, str]:
        suspect_match = re.match(r"\[(.*?)\]\s*(.*)", question)
        assert suspect_match, f"Bad question: {question}"
        suspect_name, actual_question = suspect_match.groups()
        assert suspect_name in self.hypothesis_space, f"Unrecognised: {suspect_name}"
        return suspect_name, actual_question
