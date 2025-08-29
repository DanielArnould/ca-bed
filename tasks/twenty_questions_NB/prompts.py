from textwrap import dedent

# For context (I'm getting these mixed-up):
# Questioner -> Guesser
# Answerer -> Examiner

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
    items = [h for (h, _p) in belief_state]
    items_str = "\n".join(f"- {x}" for x in items)

    asked = ""
    if history:
        asked_pairs = "; ".join([f"Q: {q}" for (q, _) in history])
        asked = f"You already asked: {asked_pairs}."

    tpl = f'''Here are all the X:
        {items_str}

        Please design a question about X and can only be answer by YES or NO. {asked} Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
        Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
        You should think about best {m} questions to response. And your answer should be(NO EXTRA TEXT):
        Question 1: Is X ...?
        YES: aaa, bbb, ...
        Count of YES: ...
        NO: ccc, ddd, ...
        Count of NO: ...
    '''
    return dedent(tpl).strip()

def get_verbalization_probability_elicitation_prompt(
    hypothesis_space: list[str], question: str
) -> str:
    items_str = "\n".join(f"- {x}" for x in hypothesis_space)

    tpl = f"""
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
    """
    return dedent(tpl).strip()


def get_targeting_prompt(hypothesis_space: list[str], history: list[tuple[str, str]]) -> str:

    return (
        dedent("""
        Note that you should guess and ask what X exactly is from now on. All possible X are in: {hypothesis_space}
        The question must start with 'Is X ..
    """)
        .format(history=bullets_history, most_likely_item=top_item)
        .strip()
    )


if __name__ == "__main__":
    print(
        get_question_generation_prompt(
            5,
            history=[("Question 1", "Yes"), ("Question 2", "No")],
            belief_state=[("dog", 0.9), ("cat", 0.05)],
        )
    )
