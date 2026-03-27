import re
from textwrap import dedent
from typing import override

from ca_bed.llm import LLM, get_response
from ca_bed.node import QuestionAnswer
from ca_bed.tasks.detective_cases.data import DetectiveCasesInstance, SuspectInformation
from ca_bed.tasks.task import TreeBasedTask


class DetectiveCasesBayesian(TreeBasedTask):
    def __init__(
        self,
        questioner_llm: LLM,
        answerer_llm: LLM,
        case: DetectiveCasesInstance,
        max_conversation_depth: int,
        n_questions: int,
        max_lookahead_depth: int,
        confidence_threshold: float,
        estimator_confidence: float,
    ) -> None:
        self.case = case

        # Determine the task answer (the murderer) and hypothesis space (suspects)
        hypothesis_space = [s["name"] for s in case["suspects"]]
        task_answer = next(s["name"] for s in case["suspects"] if s["is_murderer"])

        super().__init__(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=task_answer,
            hypothesis_space=hypothesis_space,
            max_conversation_depth=max_conversation_depth,
            n_questions=n_questions,
            max_lookahead_depth=max_lookahead_depth,
            confidence_threshold=confidence_threshold,
            estimator_confidence=estimator_confidence,
        )

    @override
    def get_id(self) -> str:
        return f"detective_cases_bayesian_{self.case['num']}"

    @override
    async def create_questions(
        self, conversation_history: list[QuestionAnswer], belief_state: dict[str, float]
    ) -> dict[str, list[str]]:
        possible_suspects = [
            entity for entity, prob in belief_state.items() if prob > 0
        ]
        prompt = build_question_generation_prompt(
            self.case, conversation_history, possible_suspects, self.n_questions
        )

        response = await get_response(prompt, self.questioner_llm)
        questions = parse_question_generation_response(response)

        return {q: ["Yes", "No"] for q in questions[: self.n_questions]}

    @override
    async def get_likelihoods(
        self, question: str, possible_answers: list[str]
    ) -> dict[str, dict[str, float]]:
        prompt = build_likelihood_prompt(self.case, question, self.hypothesis_space)

        response = await get_response(prompt, self.questioner_llm)
        likelihoods = parse_likelihood_response(response)

        return {
            "Yes": {entity: p for entity, p in likelihoods.items()},
            "No": {entity: 1.0 - p for entity, p in likelihoods.items()},
        }

    @override
    async def get_answer(self, question: str, possible_answers: list[str]) -> str:
        # Extract who the question is directed to
        target_match = re.search(r"\[Target:\s*(.*?)\]", question)
        if not target_match:
            raise ValueError(f"Question missing target format: {question}")

        target_name = target_match.group(1).strip()
        target_suspect = next(
            (
                s
                for s in self.case["suspects"]
                if s["name"].lower() in target_name.lower()
            ),
            None,
        )

        if not target_suspect:
            raise ValueError(f"Could not find suspect {target_name} in case.")

        prompt = build_answer_prompt(self.case, target_suspect, question)
        response = await get_response(prompt, self.answerer_llm)
        answer = parse_answer_response(response)
        return answer


def build_case_context(case: DetectiveCasesInstance) -> str:
    parts = [
        f"Time: {case['time']}",
        f"Location: {case['location']}",
        f"Victim: {case['victim']['name']} - {case['victim']['introduction']} (Cause of death: {case['victim']['cause_of_death']}, Weapon: {case['victim']['murder_weapon']})",
        "\nSuspects:",
    ]
    for s in case["suspects"]:
        parts.append(
            f"- {s['name']}: {s['introduction']}. "
            f"Reason at scene: {s['reason_at_scene']} "
            f"Relationship to victim: {s['relationship']} "
            f"Motive: {s['motive']} "
            f"Opportunity: {s['opportunity']} "
            f"Testimony: {s['testimony']}"
        )
    return "\n".join(parts)


def build_question_generation_prompt(
    case: DetectiveCasesInstance,
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    n_questions: int,
) -> str:
    context = build_case_context(case)
    possible_entities_str = "; ".join(entities)

    prompt_parts = [
        "You are the lead detective trying to solve a murder mystery.",
        "Your goal is to deduce the murderer among the remaining suspects.",
        f"The murderer is currently believed to be one of these: {possible_entities_str}.",
        "\n### Case Details ###",
        context,
    ]

    if conversation_history:
        prompt_parts.append("\n### Interrogation History so far ###")
        prompt_parts.extend(
            f"{idx}. Q: {qa.question}; A: {qa.answer}"
            for idx, qa in enumerate(conversation_history, start=1)
        )

    is_plural = n_questions > 1
    prompt_parts.extend(
        [
            f"\nWhat {'are' if is_plural else 'is'} {n_questions} excellent yes/no question{'s' if is_plural else ''} that you could ask to narrow down the suspect list?",
            "You MUST specify WHICH suspect you are asking the question to using the [Target: Name] format.",
            "Provide a short explanation for your reasoning. Then provide the result strictly in the format, including the double hashtag: ",
            "##Question##: [Target: <Suspect Name>] <Your yes/no question here>",
            "##Question##: [Target: <Suspect Name>] <Your yes/no question here>",
        ]
    )
    return "\n".join(prompt_parts)


def parse_question_generation_response(response: str) -> list[str]:
    question_generation_regex = r"##Question##[^:]*:[^\[]*(\[Target:.*?\]\s*.*?\?)"
    return re.findall(question_generation_regex, response)


def build_likelihood_prompt(
    case: DetectiveCasesInstance, question: str, entities: list[str]
) -> str:
    context = build_case_context(case)
    possible_entities_str = "; ".join(entities)

    return dedent(f"""\
        You are analyzing an interrogation in a murder mystery. 
        
        ### Case Details ###
        {context}

        The detective asked the following question:
        "{question}"
        
        The possible answers are "Yes" and "No". The possible true murderers are: {possible_entities_str}. 
        
        For each suspect, ASSUMING THEY ARE THE ACTUAL MURDERER, how likely is it that the targeted suspect being questioned would answer "Yes"? 
        (Keep in mind the target might be innocent and telling the truth, or the target might be the murderer and lying to protect themselves).
        
        First, provide a short explanation of your reasoning.
        Then provide the result strictly in the following format, including the double hashtag.

        ##<Suspect Name>##: <a single number between 0 and 1>
        """).strip()


def parse_likelihood_response(response: str) -> dict[str, float]:
    likelihood_regex = r"##(.*?)##[^:]*:\s*(\d+(?:\.\d+)?)"
    return {
        key.strip(): float(likelihood)
        for key, likelihood in re.findall(likelihood_regex, response, re.IGNORECASE)
    }


def build_answer_prompt(
    case: DetectiveCasesInstance,
    target_suspect: SuspectInformation,
    question: str,
) -> str:
    # Strip the [Target: Name] prefix so the answerer just sees the raw question
    clean_question = re.sub(r"\[Target:\s*.*?\]\s*", "", question)

    is_murder = target_suspect["is_murderer"]
    role_instruction = (
        "You are the MURDERER. You must lie if necessary to avoid getting caught, but try to sound consistent with your alibi."
        if is_murder
        else "You are INNOCENT. You should tell the truth based on your knowledge and relationship to the victim."
    )

    # Generate the full case context
    context = build_case_context(case)

    return dedent(f"""\
        You are roleplaying as {target_suspect["name"]} in a murder investigation.
        
        ### Case Details ###
        {context}
        
        ### Your Profile ###
        Name: {target_suspect["name"]}
        Story: {target_suspect["story"]}
        
        ### Your Secret Reality ###
        {role_instruction}
        
        The detective asks you: '{clean_question}'
        
        Answer based on your profile and secret reality. You must reply strictly with exactly "Yes" or "No".
        Provide the result strictly in the format, including the double hashtag:

        ##Answer##: <'Yes' or 'No'>
    """).strip()


def parse_answer_response(response: str) -> str:
    answer_regex = r"##Answer##[^:]*:\s*(Yes|No)"
    matches = re.findall(answer_regex, response, re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(
            f"Unexpected number of matches in '{response}'. Expected exactly one 'Yes' or 'No'."
        )
    return matches[0].capitalize()
