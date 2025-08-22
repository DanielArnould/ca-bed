"""
This module should work on unifying all prompt calls from different tasks
into one interface.
"""

from dataclasses import dataclass, field
import re

# from prompts import create_questions_prompt

"""
TODO: Create chat utilities to get LLM responses for multiple different
LLMs.

Also, make sure we can include history between two agents
"""


def ask_llm(message: str) -> str: ...


@dataclass
class Question:
    question: str
    items_yes: list[str] = field(default_factory=list)
    items_no: list[str] = field(default_factory=list)


def create_questions(
    items: list[str], previous_questions: list[str], n: int
) -> list[Question]:
    if len(items) <= 1:
        return []

    # message = create_questions_prompt(items, previous_questions, n)
    response = ask_llm(message)

    questions = []
    current_question = None

    for line in response.splitlines():
        line = line.strip()
        question_match = re.match(r".*Question\s+\d+:\s*(.+\?)", line, re.IGNORECASE)
        yes_match = re.match(r"YES:\s*(.+)", line, re.IGNORECASE)
        no_match = re.match(r"NO:\s*(.+)", line, re.IGNORECASE)

        if question_match:
            current_question = Question(question_match.group(1))
            questions.append(current_question)
            continue

        if not current_question:
            continue

        if yes_match:
            items_str = yes_match.group(1)
            current_question.items_yes.extend(
                item.strip() for item in items_str.split(",") if item.strip()
            )
        elif no_match:
            items_str = no_match.group(1)
            current_question.items_no.extend(
                item.strip() for item in items_str.split(",") if item.strip()
            )

    return questions
