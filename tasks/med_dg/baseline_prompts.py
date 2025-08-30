from textwrap import dedent

def get_questioner_prologue(problem_description: str) -> str:
    return (
        dedent("""
        You are an expert medical doctor, and your patient self-reports that: {self_report}.
        You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
        Use the ongoing conversation for context to avoid redundant questions. 
        Let us begin. Ask me the first question.
    """)
        .format(self_report=problem_description)
        .strip()
    )


def get_answer_selection_prompt(ground_truth: str, question: str) -> str:
    return (
        dedent("""
        You are the patient suffering from {target_item}, and I am the doctor. 
        I will ask you up to 6 questions, and you should answer each one truthfully based on your disease, by saying 'Yes' or 'No'. 
        Note that you must never reveal the disease until I tell it correctly. 
        If I tell the disease correctly in my question, directly respond: "You are right. I am experiencing {target_item}."
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
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases:
            {belief}

            Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
            Format your response in this structure:
            1. <Question 1>
            2. <Question 2>
            ...
            n. <Question n>
    """)
            .format(history=bullets_history, belief=bullets_pruned_hypothesis, num_questions=m)
            .strip()
        )

    return (
        dedent("""
        These are the questions you've asked to the patient so far:
        {history}

        Based on our current beliefs, the patient is most likely suffering from one of the following diseases:
        {belief}

        Your task is to generate {num_questions} *excellent* yes/no questions to ask next. The best questions are those that will help distinguish between these likely possibilities.
        Format your response in this structure:
        1. <Question 1>
        2. <Question 2>
        ...
        n. <Question n>
    """)
        .format(history=bullets_history, belief=bullets_pruned_hypothesis, num_questions=m)
        .strip()
    )


def get_verbalization_probability_elicitation_prompt(
    problem_description: str, belief_state: list[tuple[str, float]], question: str
) -> str:
    pruned_hypothesis = [item for item, prob in belief_state if prob > 0]
    items_str = "\n".join(f"- {x}" for x in pruned_hypothesis)

    return (
        dedent("""
        You are an expert medical doctor, and your patient self-reports that: {self_report}. 

        Here are all the possible diseases that the patient may suffer from:
        {items_str}

        Classify the disease based on this single yes/no question:
        Question: "{question}"

        If the answer would be YES when the patient is indeed suffering from that disease, put it in YES; otherwise put it in NO.
        Use the disease names exactly as listed. Cover all diseases exactly once (no omissions, no duplicates).

        Return exactly in this format (no extra text, JUST WHAT I'M FORMATTING BELOW):

        Question 1: <question>
        YES: aaaa, bbbb, ...
        Count of YES: <integer>
        NO: cccc, dddd, ...
        Count of NO: <integer>
    """)
        .format(self_report=problem_description, items_str=items_str, question=question)
        .strip()
    )

def get_targeting_prompt() -> str:
    return dedent("""
        Note that you should point out and ask what disease the patient suffers from now.
        Refer to the past conversation regarding the patient's symptoms. Never repeat previously asked questions.
        Ask the question with following the format: 'Are you experiencing [disease name]?'
    """).strip()