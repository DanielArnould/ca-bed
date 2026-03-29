import asyncio
import pickle
import re
from pathlib import Path

from ca_bed.llm import LLM, get_response
from ca_bed.tasks.twenty_questions.data import TWENTY_QUESTIONS_ENTITIES
from tqdm.asyncio import tqdm


def build_prediction_prompt(conversation_history: list[tuple[str, str]]) -> str:
    possible_entities_str = "; ".join(TWENTY_QUESTIONS_ENTITIES)
    prompt_parts = [
        f"You are an expert at the game of 20 Questions. The possible entities are: {possible_entities_str}.",
        "Below is a transcript of a game that was just played. Based on the questions and answers, your task is to deduce the secret entity.",
        "\n### Game Transcript ###",
    ]

    for idx, (q, a) in enumerate(conversation_history, start=1):
        prompt_parts.append(f"{idx}. Q: {q} | A: {a}")

    prompt_parts.extend(
        [
            "\nProvide a brief explanation of your reasoning, then provide your final guess strictly in the following format, including the double hashtags:",
            "##Guess##: <Your predicted entity here>",
        ]
    )

    return "\n".join(prompt_parts)


def parse_prediction_response(response: str) -> str:
    guess_regex = r"##Guess##\s*:\s*(.*)"
    matches = re.findall(guess_regex, response, re.IGNORECASE)
    if not matches:
        return "Unknown (Failed to parse)"
    # Strip whitespace and trailing punctuation just in case
    return matches[-1].strip(" .")


async def analyze_game(file_path: Path, llm: LLM) -> bool | None:
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        conversation_history = [
            (node.parent.question, node.answer) for node in obj.final_path[1:]
        ]
        true_answer = obj.task.task_answer

        prompt = build_prediction_prompt(conversation_history)
        response = await get_response(prompt, llm)
        prediction = parse_prediction_response(response)

        return prediction.lower() in true_answer.lower()

    except Exception:
        return None


async def main():
    llm = LLM("deepseek-reasoner")

    results_dir = Path(r"results/TwentyQuestions/Direct/")
    pickle_files = list(results_dir.rglob("*.pickle"))

    if not pickle_files:
        print(f"No .pickle files found in {results_dir}")
        return

    print(f"Analyzing {len(pickle_files)} games...")

    tasks = [analyze_game(file_path, llm) for file_path in pickle_files]

    results = await tqdm.gather(*tasks, desc="Analyzing Games")
    valid_results = [res for res in results if res is not None]

    if not valid_results:
        print("No valid games could be analyzed due to errors.")
        return

    correct_count = sum(valid_results)
    total_valid = len(valid_results)
    accuracy = (correct_count / total_valid) * 100

    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_valid} games)")


if __name__ == "__main__":
    asyncio.run(main())
