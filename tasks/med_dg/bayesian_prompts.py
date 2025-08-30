from textwrap import dedent

def get_questioner_prologue(problem_description: str, hypothesis_space: list[str]) -> str:
    bullets_hypothesis = "\n".join(f"- {h}" for h in hypothesis_space)
    return (
        dedent("""
        You are an expert medical doctor, and your patient self-reports that: {self_report}. 
               
        The patient is suffering from one of the following possible diseases:      
        {hypothesis}
    
        You should ask your patient questions in English about symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers from. 
        Use the ongoing conversation for context to avoid redundant questions. 
        Let us begin. Ask me the first question.
    """)
        .format(self_report=problem_description, hypothesis=bullets_hypothesis)
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
    bullets_belief = "\n".join(
        f"- Disease: {b[0]}; Probability: {b[1]}" for b in belief_state
    )

    if len(bullets_history) == 0:
        return (
            dedent("""
            Based on our current beliefs, the patient is most likely suffering from one of the following diseases, which are listed along with their probabilities:
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
        These are the questions you've asked to the patient so far:
        {history}

        Based on our current beliefs, the patient is most likely suffering from one of the following diseases, which are listed along with their probabilities:
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
    problem_description:str, hypothesis_space: list[str], question: str
) -> str:
    bullets = "\n".join(f"- {h}" for h in hypothesis_space)

    return (
        dedent("""
        You are an expert medical doctor, and your patient self-reports that: {self_report}. 
        You need to estimate the probability of a "Yes" answer for a list of different potential diseases a patient is suffering from, given a single question.

        The question is:
        "{candidate_question}"

        For each of the following diseases, estimate the probability that a patient would answer "Yes" to the question above if they were indeed suffering from that disease.

        Diseases:
        {hypothesis}

        Please provide your response ONLY as a single JSON object. The keys should be the disease and the values should be the estimated probability (a float between 0.0 and 1.0).

        Example format for the question "Do you have a fever?" (THE FOLLOWING PROBABILITIES ARE HYPOTHETICAL):
        {{
        "Flu": 0.33,
        "Penumonia": 0.33,
        "Rubella": 0.33,
        "Anemia": 0.0
        }}
    """)
        .format(self_report=problem_description, hypothesis=bullets, candidate_question=question)
        .strip()
    )