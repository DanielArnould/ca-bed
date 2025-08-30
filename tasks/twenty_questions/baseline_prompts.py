from textwrap import dedent


def get_questioner_prologue() -> str:
    return dedent("""
        You are an expert player of the 20 Questions game. Your goal is to guess a secret object, X. I will be impersonating the secret object, X.
        You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
        Let us begin. Ask me the first question.
    """).strip()


def get_answer_selection_prompt(ground_truth: str, question: str) -> str:
    return (
        dedent("""
        You are an expert player of the 20 Questions game. Your goal is to impersonate the secret object, X. I will be trying to guess the secret object, X. X is {target_item}.
        I will ask up to 20 questions and you should answer each one truthfully based on being X, by saying 'Yes' or 'No'.
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
    pruned_hypothesis = [item for item, prob in belief_state if prob > 0]
    bullets_pruned_hypothesis = "\n".join(f"- {item}" for item in pruned_hypothesis)

    if len(bullets_history) == 0:
        return (
            dedent("""
            Based on our current beliefs, the secret object is most likely one of the following items:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
    """)
            .format(
                history=bullets_history,
                belief=bullets_pruned_hypothesis,
                num_questions=m,
            )
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
        .format(
            history=bullets_history, belief=bullets_pruned_hypothesis, num_questions=m
        )
        .strip()
    )


def get_verbalization_probability_elicitation_prompt(
    belief_state: list[tuple[str, float]], question: str
) -> str:
    pruned_hypothesis = [item for item, prob in belief_state if prob > 0]
    items_str = "\n".join(f"- {x}" for x in pruned_hypothesis)

    return (
        dedent("""
        Here are all the X:
        {items_str}

        Classify the X based on this single yes/no question:
        Question: "{question}"

        If the answer would be YES when that X is the secret object, put it in YES; otherwise put it in NO.
        Use the item strings exactly as listed. Cover all items exactly once (no omissions, no duplicates).

        Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

        Question 1: <question>
        YES: aaaa, bbbb, ...
        Count of YES: <integer>
        NO: cccc, dddd, ...
        Count of NO: <integer>
    """)
        .format(items_str=items_str, question=question)
        .strip()
    )


def get_targeting_prompt() -> str:
    return dedent("""
        Note that you should guess and ask what X exactly is from now on.
        The question must start with 'Is X ..
    """).strip()
