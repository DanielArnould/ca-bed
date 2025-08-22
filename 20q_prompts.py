"""
##################################################
#                   DEPRECATED                   #
##################################################

Only remains here as reference for future prompt functions
"""


def create_questions_prompt(
    items: list[str], previous_questions: list[str], n: int
) -> str:
    do_not_ask_previous_questions = (
        "(The question should not be '" + "' or '".join(previous_questions) + "')"
        if previous_questions
        else ""
    )
    formatted_items = ", ".join(items)

    return f"""Here are all the X:
    {formatted_items}

    Please design a question about X and can only be answer by YES or NO. {do_not_ask_previous_questions} Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
    Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
    You should think about best {n} questions to response. And your answer should be:
    Question 1: Is X ...?
    YES: aaa, bbb, ...
    Count of YES: ...
    NO: ccc, ddd, ...
    Count of NO: ...
    """


def target_question(item: str) -> str:
    return f"Is X a '{item}'?"


def target_prologue(items: list[str]) -> str:
    formatted_items = ", ".join(items)
    return f"""Note that you should guess and ask what X exactly is from now on. X is possible a:
    {formatted_items}, or other.
    The question must start with 'Is X ...'"""


def guesser_prologue() -> str:
    return """Let us play the game of 20 questions. I am impersonating the thing, X. You will ask me up to 20 questions which start with 'Is X' and can only be answered by yes or no, and I will answer each one truthfully based on being X.
    Let us begin. Ask me the first question.
    """


def examiner_prologue(item: str) -> str:
    return f"""Let us play the game of 20 questions. You are the answerer and I am the guesser. X is '{item}'. 
    I will ask you up to 20 questions and you should answer each one truthfully based on being X. 
    If I guess correctly what X is, answer me "You guessed it. X is '{item}'."
    You must NEVER directly tell me what X is.
    Let us begin. Here is my first question.
    """
