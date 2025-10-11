from textwrap import dedent
from typing import override

from tenacity import retry, stop_after_attempt
from models import LLMRequestSession, query_llm
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.detective_cases.common import get_case_background, parse_question
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.task import (
    Task,
    parse_answer,
    parse_binary_questions,
    parse_categorical_likelihoods,
    parse_uniform_probabilities,
)


class Baseline(Task):
    instance: DetectiveCasesInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: DetectiveCasesInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
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
            confidence_threshold=1.0,
            hypothesis_space=[suspect["name"] for suspect in self.instance["suspects"]],
        )

    def __str__(self) -> str:
        return f"Detective Cases (Baseline): {self.task_answer=} {self.max_question_nodes=} {self.max_lookahead_depth=} {self.max_conversation_depth=} {self.confidence_threshold=} {self.hypothesis_space=}"

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
            {get_case_background(self.instance)}

            The investigation focuses on {len(self.hypothesis_space)} suspects, one of whom is the true murderer:
            {suspects_info}

            ### Task
            - Select one or more suspects from {self.hypothesis_space} you wish to investigate.
            - You must select at least one.

            ### Response Format
            One line per suspect:
            <number>. <Suspect Name>

            ### Example
            1. Suspect A 
            2. Suspect B  
            3. Suspect C  
            4. Suspect D  
            5. Suspect E  
            """).strip()

        output = await query_llm(prompt, self.questioner_session)
        priors = parse_uniform_probabilities(output)

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
            {get_case_background(self.instance)}
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
            f"- Suspect: {hypo}" for hypo, _ in current_node.belief_state.items()
        )
        parts.append(
            dedent(f"""\
            Based on the current belief state, these are the candidate suspects for being the murderer:
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
            1. [Alice] Were you outside at 12:00PM? 
            2. [Bob] Did you have access to the murder weapon?
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
        answerer_name, actual_question = parse_question(self.hypothesis_space, question)

        suspects = [s for s in self.instance["suspects"] if s["name"] in hypotheses]
        assert len(suspects) > 0, f"No matching suspect found in question: {question}"

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
            {get_case_background(self.instance)}

            ### Suspects
            {suspects_info}

            ### Question to {answerer_name}
            "{actual_question}"

            ### {len(answers)} Possible Answers
            {answers}

            ### Task
            - Interpret the question and possible answers.  
            - For each suspect, assume they are the murderer and decide whether {answerer_name} would most likely say 'Yes' or 'No'.  
            - Assign each suspect to exactly one of 'Yes' or 'No' (no omissions, no duplicates).  
            - Use the suspect names exactly as given.
            - Display the answers exactly in the order as given.

            ### Response Format

            Yes: Suspect_1, Suspect_2, ...
            Count of 'Yes': <integer>
            No: Suspect_3, Suspect_4, ...
            Count of 'No': <integer>

            ### Example

            Yes: Ms. Alice, Dr. Bob
            Count of 'Yes': 2
            No: Mr. Charlie
            Count of 'No': 1

            Do not include commentary or explanations. Return only the formatted response.
        """)

        # Query LLM
        output = await query_llm(prompt, self.questioner_session)
        likelihoods = parse_categorical_likelihoods(output, possible_answers=answers)
        return {
            likelihood.hypothesis: {
                ans: prob
                for ans, prob in zip(answers, likelihood.likelihoods, strict=True)
            }
            for likelihood in likelihoods
        }

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        suspect_name, question = parse_question(
            self.hypothesis_space, current_node.question
        )

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
