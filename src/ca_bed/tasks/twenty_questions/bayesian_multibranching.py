import re
from textwrap import dedent
from typing import override

from ca_bed.llm import get_response
from ca_bed.node import QuestionAnswer
from ca_bed.tasks.task import TreeBasedTask


class TwentyQuestionsBayesianMultibranching(TreeBasedTask):
    @override
    def get_id(self) -> str:
        return f"20q_bayesian_multibranching_{self.task_answer}"

    @override
    async def create_questions(
        self, conversation_history: list[QuestionAnswer], belief_state: dict[str, float]
    ) -> dict[str, list[str]]:
        possible_entities = [
            entity for entity, prob in belief_state.items() if prob > 0
        ]
        prompt = build_question_generation_prompt(
            conversation_history, possible_entities, self.n_questions
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
            question, possible_answers, self.hypothesis_space
        )
        response = await get_response(prompt, self.questioner_llm)
        parsed_likelihoods = parse_likelihood_response(response)

        # Structure: { "A": {"Apple": 0.9, "Banana": 0.0}, "B": {...} }
        likelihoods = {ans: {} for ans in possible_answers}
        for entity, letter, prob in parsed_likelihoods:
            # Clean up letter casing just in case (e.g., "A" vs "a")
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
        prompt = build_answer_prompt(self.task_answer, question, possible_answers)
        response = await get_response(prompt, self.answerer_llm)
        answer = parse_answer_response(response)

        return answer


def build_question_generation_prompt(
    conversation_history: list[QuestionAnswer],
    entities: list[str],
    n_questions: int,
) -> str:
    possible_entities_str = "; ".join(entities)
    prompt_parts = [
        "You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X.",
        f"The secret entity X is one of these: {possible_entities_str}.",
    ]

    if conversation_history:
        prompt_parts.append(" The game so far:")
        prompt_parts.extend(
            f"{idx}. Q: {qa.question}\n   A: {qa.answer}"
            for idx, qa in enumerate(conversation_history, start=1)
        )

    is_plural = n_questions > 1
    prompt_parts.extend(
        [
            f"What {'are' if is_plural else 'is'} {n_questions} excellent multiple-choice question{'s' if is_plural else ''} that you could ask next?",
            "For each question, provide 2 to 5 mutually exclusive possible answers.",
            "Provide a short explanation for your reasoning. Then provide the result strictly in the following format, including the exact double hashtag tags:",
            "##Question##: <The question text>",
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
        q_match = re.match(r"##Question##[^:]*:\s*(.*)", line)
        if q_match:
            # Save the previous question block before starting a new one
            if current_question and current_letters:
                full_context = current_question + "\n".join(current_options_text)
                questions.append((full_context, current_letters))

            current_question = q_match.group(1).strip()
            current_options_text = []
            current_letters = []
            continue

        # 2. Match Explicitly Tagged Options
        opt_match = re.match(r"##Option\s+([A-Z])##[^:]*:\s*(.*)", line)
        if opt_match and current_question:
            letter = opt_match.group(1).upper()
            opt_text = opt_match.group(2).strip()

            current_letters.append(letter)
            # Standardize the text format ourselves (e.g., "A) Option text")
            current_options_text.append(f"{letter}) {opt_text}")

    # Save the final question block in the loop
    if current_question and current_letters:
        full_context = current_question + "\n".join(current_options_text)
        questions.append((full_context, current_letters))

    return questions


def build_likelihood_prompt(
    question: str, possible_answers: list[str], entities: list[str]
) -> str:
    possible_entities_str = "; ".join(entities)
    valid_letters_str = ", ".join(possible_answers)

    return dedent(f"""\
            You are analysing a game of 20 questions. The question and options were:
            "{question}"
            
            The valid answer letters are: {valid_letters_str}. 
            The possible entities are: {possible_entities_str}. 
            
            For each entity, assuming it was the answer, how likely is it that the answerer would 
            choose each specific option letter?
            
            First, provide a short explanation of your reasoning.
            Then provide the result strictly in the following format, including the double hashtag 
            with the Entity and the Letter separated by a pipe (|):

            ##Entity|Letter##: <a single number between 0 and 1>
            
            Example:
            ##Apple|A##: 0.8
            ##Apple|B##: 0.1
            """).strip()


def parse_likelihood_response(response: str) -> list[tuple[str, str, float]]:
    # Extracts the Entity, the Letter, and the probability score
    likelihood_regex = r"##(.*?)\|([A-Z])##[^:]*:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(likelihood_regex, response)

    return [
        (entity.strip(), letter.strip().upper(), float(prob))
        for entity, letter, prob in matches
    ]


def build_answer_prompt(
    secret_entity: str, question: str, possible_answers: list[str]
) -> str:
    valid_letters_str = ", ".join(possible_answers)
    return dedent(f"""\
        You are a player of the 20 Questions game. Your goal is
        to impersonate the secret entity, X. X is {secret_entity}. 
        
        You have just been asked the following multiple-choice question:
        {question}
        
        Answer truthfully based on what X is, choosing EXACTLY ONE of the valid option letters ({valid_letters_str}).
        Provide the result strictly in the format, including the double hashtag:

        ##Answer##: <A single letter representing your choice>
    """).strip()


def parse_answer_response(response: str) -> str:
    answer_regex = r"##Answer##[^:]*:\s*([A-Z])"
    match = re.search(answer_regex, response, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    raise ValueError(f"Could not find valid ##Answer## letter in '{response}'")
