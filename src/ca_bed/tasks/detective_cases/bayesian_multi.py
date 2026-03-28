import re
from textwrap import dedent
from typing import override

from ca_bed.llm import LLM, get_response
from ca_bed.node import QuestionAnswer
from ca_bed.tasks.detective_cases.data import DetectiveCasesInstance, SuspectInformation
from ca_bed.tasks.task import TreeBasedTask


class DetectiveCasesBayesianMultibranching(TreeBasedTask):
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
        return (
            f"detective_cases_bayesian_multibranching_{self.case.get('num', 'unknown')}"
        )

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
        questions_and_answers = parse_question_generation_response(response)

        return {
            question: answers
            for (question, answers) in questions_and_answers[: self.n_questions]
        }

    @override
    async def get_likelihoods(
        self, question: str, possible_answers: list[str]
    ) -> dict[str, dict[str, float]]:
        prompt = build_likelihood_prompt(
            self.case, question, possible_answers, self.hypothesis_space
        )
        response = await get_response(prompt, self.questioner_llm)
        parsed_likelihoods = parse_likelihood_response(response)

        # Structure: { "A": {"Alice": 0.9, "Bob": 0.0}, "B": {...} }
        likelihoods = {ans: {} for ans in possible_answers}
        for entity, letter, prob in parsed_likelihoods:
            # Clean up letter casing just in case
            matched_letter = next(
                (a for a in possible_answers if a.upper() == letter.upper()), letter
            )
            if matched_letter in likelihoods:
                likelihoods[matched_letter][entity] = prob

        # Normalize the probabilities so they sum to 1.0 for each entity
        for entity in self.hypothesis_space:
            total_prob = sum(
                likelihoods[ans].get(entity, 0.0) for ans in possible_answers
            )
            for ans in possible_answers:
                if total_prob > 0:
                    likelihoods[ans][entity] = (
                        likelihoods[ans].get(entity, 0.0) / total_prob
                    )
                else:
                    likelihoods[ans][entity] = 1.0 / len(possible_answers)

        return likelihoods

    @override
    async def get_answer(self, question: str, possible_answers: list[str]) -> str:
        # Extract who the question is directed to
        target_match = re.search(r"\[Target:\s*(.*?)\]", question, re.IGNORECASE)
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

        prompt = build_answer_prompt(
            self.case, target_suspect, question, possible_answers
        )
        response = await get_response(prompt, self.answerer_llm)
        answer = parse_answer_response(response)

        # Fallback safeguard
        if answer not in possible_answers:
            answer = possible_answers[0]

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
            f"{idx}. Q: {qa.question}\n   A: {qa.answer}"
            for idx, qa in enumerate(conversation_history, start=1)
        )

    is_plural = n_questions > 1
    prompt_parts.extend(
        [
            f"\nWhat {'are' if is_plural else 'is'} {n_questions} excellent multiple-choice question{'s' if is_plural else ''} that you could ask to narrow down the suspect list?",
            "You MUST specify WHICH suspect you are asking the question to using the [Target: Name] format.",
            "For each question, provide 2 to 5 mutually exclusive possible answers.",
            "Provide a short explanation for your reasoning. Then provide the result strictly in the following format, including the exact double hashtag tags:",
            "##Question##: [Target: <Suspect Name>] <The question text>",
            "##Option A##: <First Option text>",
            "##Option B##: <Second Option text>",
            "##Option C##: <Third Option text>",
            "",
            "If asking multiple questions, simply start a new ##Question## tag.",
        ]
    )
    return "\n".join(prompt_parts)


def parse_question_generation_response(response: str) -> list[tuple[str, list[str]]]:
    questions: list[tuple[str, list[str]]] = []
    current_question = None
    current_options_text = []
    current_letters = []

    # Process line-by-line to completely avoid multiline regex failures
    for line in response.splitlines():
        line = line.strip()

        # 1. Match Question
        q_match = re.match(
            r"##Question##[^:]*:[^\[]*(\[Target:.*?\]\s*.*?\?)", line, re.IGNORECASE
        )
        if q_match:
            # Save the previous question block before starting a new one
            if current_question and current_letters:
                full_context = current_question + "\n" + "\n".join(current_options_text)
                questions.append((full_context, current_letters))

            current_question = q_match.group(1).strip()
            current_options_text = []
            current_letters = []
            continue

        # 2. Match Explicitly Tagged Options
        opt_match = re.match(r"##Option\s+([A-Z])##[^:]*:\s*(.*)", line, re.IGNORECASE)
        if opt_match and current_question:
            letter = opt_match.group(1).upper()
            opt_text = opt_match.group(2).strip()

            current_letters.append(letter)
            current_options_text.append(f"{letter}) {opt_text}")

    # Save the final question block in the loop
    if current_question and current_letters:
        full_context = current_question + "\n" + "\n".join(current_options_text)
        questions.append((full_context, current_letters))

    return questions


def build_likelihood_prompt(
    case: DetectiveCasesInstance,
    question: str,
    possible_answers: list[str],
    entities: list[str],
) -> str:
    context = build_case_context(case)
    possible_entities_str = "; ".join(entities)
    valid_letters_str = ", ".join(possible_answers)

    return dedent(f"""\
        You are analyzing an interrogation in a murder mystery. 
        
        ### Case Details ###
        {context}

        The detective asked the following multiple-choice question:
        "{question}"
        
        The valid answer letters are: {valid_letters_str}. 
        The possible true murderers are: {possible_entities_str}. 
        
        For each suspect, ASSUMING THEY ARE THE ACTUAL MURDERER, how likely is it that the targeted suspect being questioned would choose each specific option letter? 
        (Keep in mind the target might be innocent and telling the truth, or the target might be the murderer and lying to protect themselves).
        
        First, provide a short explanation of your reasoning.
        Then provide the result strictly in the following format, including the double hashtag 
        with the Suspect and the Letter separated by a pipe (|):

        ##Suspect Name|Letter##: <a single number between 0 and 1>
        
        Example:
        ##Alice|A##: 0.8
        ##Alice|B##: 0.1
        """).strip()


def parse_likelihood_response(response: str) -> list[tuple[str, str, float]]:
    # Extracts the Entity, the Letter, and the probability score
    likelihood_regex = r"##(.*?)\|([A-Z])##[^:]*:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(likelihood_regex, response, re.IGNORECASE)

    return [
        (entity.strip(), letter.strip().upper(), float(prob))
        for entity, letter, prob in matches
    ]


def build_answer_prompt(
    case: DetectiveCasesInstance,
    target_suspect: SuspectInformation,
    question: str,
    possible_answers: list[str],
) -> str:
    # Strip the [Target: Name] prefix so the answerer just sees the raw question and options
    clean_question = re.sub(r"\[Target:\s*.*?\]\s*", "", question, flags=re.IGNORECASE)
    valid_letters_str = ", ".join(possible_answers)

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
        
        The detective asks you the following multiple-choice question:
        {clean_question}
        
        Answer truthfully based on your profile and secret reality, choosing EXACTLY ONE of the valid option letters ({valid_letters_str}).
        Provide the result strictly in the format, including the double hashtag:

        ##Answer##: <A single letter representing your choice>
    """).strip()


def parse_answer_response(response: str) -> str:
    answer_regex = r"##Answer##[^:]*:\s*([A-Z])"
    match = re.search(answer_regex, response, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    raise ValueError(f"Could not find valid ##Answer## letter in '{response}'")
