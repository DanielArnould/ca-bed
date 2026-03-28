import re
from textwrap import dedent
from typing import override

from ca_bed.llm import LLM, get_response
from ca_bed.node import QuestionAnswer
from ca_bed.tasks.detective_cases.data import DetectiveCasesInstance, SuspectInformation
from ca_bed.tasks.task import DirectTask, Prediction, Question


class DetectiveCasesDirect(DirectTask):
    def __init__(
        self,
        questioner_llm: LLM,
        answerer_llm: LLM,
        case: DetectiveCasesInstance,
        max_conversation_depth: int,
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
        )

    @override
    def get_id(self) -> str:
        return f"detective_cases_direct_{self.case['num']}"

    @override
    async def query_questioner(
        self, conversation_history: list[QuestionAnswer]
    ) -> Question | Prediction:
        prompt = build_questioner_prompt(
            self.case,
            conversation_history,
            self.hypothesis_space,
            self.max_conversation_depth,
        )

        response = await get_response(prompt, self.questioner_llm)
        return parse_questioner_response(response)

    @override
    async def query_answerer(self, question: str) -> str:
        # Extract who the question is directed to
        target_match = re.search(r"\[Target:\s*(.*?)\]", question, re.IGNORECASE)
        if not target_match:
            raise ValueError(f"Question missing target format: {question}")

        target_name = target_match.group(1).strip()
        target_suspect = next(
            (
                s
                for s in self.case["suspects"]
                if s["name"].lower() == target_name.lower()
            ),
            None,
        )

        if not target_suspect:
            raise ValueError(f"Could not find suspect {target_name} in case.")

        prompt = build_answerer_prompt(self.case, target_suspect, question)

        response = await get_response(prompt, self.answerer_llm)
        return parse_answerer_response(response)


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


def build_questioner_prompt(
    case: DetectiveCasesInstance,
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    max_conversation_depth: int,
) -> str:
    context = build_case_context(case)
    possible_entities_str = "; ".join(entities)

    prompt_parts = [
        "You are the lead detective trying to solve a murder mystery.",
        "Your goal is to deduce the murderer among the remaining suspects.",
        f"The murderer is exactly one of these: {possible_entities_str}.",
        "\n### Case Details ###",
        context,
    ]

    if conversation_history:
        prompt_parts.append("\n### Interrogation History so far ###")
        prompt_parts.extend(
            f"{idx}. Q: {qa.question}; A: {qa.answer}"
            for idx, qa in enumerate(conversation_history, start=1)
        )

    if len(conversation_history) >= max_conversation_depth - 3:
        prompt_parts.append(
            "\nYou are running out of questions. You must now make a prediction instead of asking a question."
        )
    else:
        prompt_parts.append(
            "\nYou can either ask an excellent yes/no question to narrow down the suspect list, or make a prediction of who the murderer is."
        )

    prompt_parts.extend(
        [
            "First, provide a short explanation for your reasoning. Then provide the result strictly in one of the following formats, including the double hashtag:",
            "",
            "If asking a question, you MUST specify WHICH suspect you are asking:",
            "##Question##: [Target: <Suspect Name>] <Your yes/no question here>",
            "",
            "If making a prediction of who the murderer is:",
            "##Prediction##: <The exact name of the murderer here>",
        ]
    )

    return "\n".join(prompt_parts)


def parse_questioner_response(response: str) -> Question | Prediction:
    question_regex = r"##Question##[^:]*:[^\[]*(\[Target:.*?\]\s*.*?\?)"
    prediction_regex = r"##Prediction##[^#]*:\s*(.*)"

    question_match = re.search(question_regex, response, re.IGNORECASE)
    prediction_match = re.search(prediction_regex, response, re.IGNORECASE)

    if question_match:
        return Question(question_match.group(1).strip())
    elif prediction_match:
        return Prediction(prediction_match.group(1).strip())
    else:
        raise RuntimeError(
            f"Response does not match expected structure in '{response}'"
        )


def build_answerer_prompt(
    case: DetectiveCasesInstance,
    target_suspect: SuspectInformation,
    question: str,
) -> str:
    # Strip the [Target: Name] prefix so the answerer just sees the raw question
    clean_question = re.sub(r"\[Target:\s*.*?\]\s*", "", question, flags=re.IGNORECASE)

    is_murder = target_suspect["is_murderer"]
    role_instruction = (
        "You are the MURDERER. You must lie if necessary to avoid getting caught, but try to sound consistent with your alibi."
        if is_murder
        else "You are INNOCENT. You should tell the truth based on your knowledge and relationship to the victim."
    )

    context = build_case_context(case)

    return dedent(f"""\
        You are roleplaying as {target_suspect["name"]} in a murder investigation.
        
        ### Case Details ###
        {context}
        
        ### Your Profile ###
        Name: {target_suspect["name"]}
        Relationship to victim: {target_suspect["relationship"]}
        Motive: {target_suspect["motive"]}
        Opportunity: {target_suspect["opportunity"]}
        What you told the police: {target_suspect["testimony"]}
        
        ### Your Secret Reality ###
        {role_instruction}
        
        The detective asks you: '{clean_question}'
        
        Answer based on your profile and secret reality. You must reply strictly with exactly "Yes" or "No".
        Provide the result strictly in the format, including the double hashtag:

        ##Answer##: <'Yes' or 'No'>
    """).strip()


def parse_answerer_response(response: str) -> str:
    answer_regex = r"##Answer##[^:]*:\s*(Yes|No)"
    matches = re.findall(answer_regex, response, re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(
            f"Unexpected number of matches in '{response}'. Expected exactly one 'Yes' or 'No'."
        )
    return matches[0].capitalize()
