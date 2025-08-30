from textwrap import dedent


def get_questioner_prologue(hypothesis_space: list[str]) -> str:
    bullets = "\n".join(f"- {h}" for h in hypothesis_space)
    return (
        dedent("""
        You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X. X is possibly one of the following:
        {hypothesis}
        You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
        Let us begin. Ask me the first question.
    """)
        .format(hypothesis=bullets)
        .strip()
    )


def get_answer_selection_prompt(ground_truth: str, question: str) -> str:
    return (
        dedent("""
        You are an expert player of the 20 Questions game. Your goal is to impersonate the secret object, X. I will be trying to guess the secret object, X. X is {target_item}.
        I will ask up to 20 questions and you should answer each one truthfully based on being X, by saying 'Yes' or 'No'. Note that you must never reveal X, until I guess it correctly.
        If I guess X correctly in my question, directly respond 'You guessed it. X is {target_item}.' instead of saying 'Yes'.
        Let us begin. Here is my question:
        {question}
    """)
        .format(target_item=ground_truth, question=question)
        .strip()
    )


def get_question_generation_prompt(
    m: int, history: list[tuple[str, str]], belief_state: list[tuple[str, float]]
) -> str:
    bullets_history = "\n".join(f"- Q: {h[0]}; A: {h[1]}" for h in history)
    bullets_belief = "\n".join(
        f"- X: {b[0]}; Probability: {b[1]}" for b in belief_state
    )

    if len(bullets_history) == 0:
        return (
            dedent("""
            Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
    """)
            .format(history=bullets_history, belief=bullets_belief, num_questions=m)
            .strip()
        )

    return (
        dedent("""
        The game has proceeded as follows:
        {history}

        Based on our current beliefs, the secret object is most likely one of the following items, which are listed along with their probabilities:
        {belief}

        Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
        Format your response in this structure:
        1. <Question 1>
        2. <Question 2>
        ...
        n. <Question n>
    """)
        .format(history=bullets_history, belief=bullets_belief, num_questions=m)
        .strip()
    )


def get_verbalization_probability_elicitation_prompt(
    hypothesis_space: list[str], question: str
) -> str:
    bullets = "\n".join(f"- {h}" for h in hypothesis_space)

    return (
        dedent("""
        Assume you are playing a game of 20 Questions. You need to estimate the probability of a "Yes" answer for a list of different potential secret objects, given a single question.

        The question is:
        "{candidate_question}"

        For each of the following items, estimate the probability that a person would answer "Yes" to the question above if that item were the secret object.

        Items:
        {hypothesis}

        Please provide your response ONLY as a single JSON object. The keys should be the item names and the values should be the estimated probability (a float between 0.0 and 1.0).

        Example format for the question "Is it an animal?":
        {{
        "Dog": 0.99,
        "Cookie": 0.0,
        "Paint": 0.0,
        "Hat": 0.0
        }}
    """)
        .format(hypothesis=bullets, candidate_question=question)
        .strip()
    )
