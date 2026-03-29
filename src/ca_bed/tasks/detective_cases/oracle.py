import asyncio
import pickle
import re
from pathlib import Path

from tqdm.asyncio import tqdm

from ca_bed.llm import LLM, get_response
from ca_bed.tasks.detective_cases.common import build_case_context


def build_prediction_prompt(case, conversation_history: list[tuple[str, str]]) -> str:
    """Builds the prompt asking the LLM to deduce the murderer."""
    case_context = build_case_context(case)

    prompt_parts = [
        "You are an expert detective analyzing an interrogation transcript to solve a murder mystery.",
        "\n### Case Details ###",
        case_context,
        "\n### Interrogation Transcript ###",
        "Below is the transcript of the interrogation. Based on the case details and these questions and answers, your task is to deduce who the murderer is.",
    ]

    for idx, (q, a) in enumerate(conversation_history, start=1):
        prompt_parts.append(f"{idx}. Q: {q} | A: {a}")

    prompt_parts.extend(
        [
            "\nProvide a brief explanation of your reasoning, then provide your final guess strictly in the following format, including the double hashtags:",
            "##Guess##: <Your predicted murderer's name here>",
        ]
    )

    return "\n".join(prompt_parts)


def parse_prediction_response(response: str) -> str:
    guess_regex = r"##Guess##\s*:\s*(.*)"
    matches = re.findall(guess_regex, response, re.IGNORECASE)
    if not matches:
        return "Unknown (Failed to parse)"
    return matches[-1].strip(" .")


async def analyze_game(file_path: Path, llm: LLM) -> bool | None:
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        conversation_history = [
            (node.parent.question, node.answer) for node in obj.final_path[1:]
        ]

        # Extract case-specific data from the saved task
        case = obj.task.case
        true_answer = obj.task.task_answer

        prompt = build_prediction_prompt(case, conversation_history)
        response = await get_response(prompt, llm)
        prediction = parse_prediction_response(response)

        # Flexible matching for names (e.g., 'Smith' in 'Mr. Smith' or vice versa)
        is_correct = (
            prediction.lower() in true_answer.lower()
            or true_answer.lower() in prediction.lower()
        )

        return is_correct

    except Exception as e:
        # print(f"Error on {file_path.name}: {e}") # Uncomment to debug corrupted files
        return None


async def main():
    llm = LLM("deepseek-reasoner")

    results_dir = Path(r"results/DetectiveCasesNew/UoT_D2_1.0_1.0/")
    pickle_files = list(results_dir.rglob("*.pickle"))

    if not pickle_files:
        print(f"No .pickle files found in {results_dir}")
        return

    print(f"Analyzing {len(pickle_files)} detective games...")

    tasks = [analyze_game(file_path, llm) for file_path in pickle_files]

    results = await tqdm.gather(*tasks, desc="Analyzing Games")
    valid_results = [res for res in results if res is not None]

    if not valid_results:
        print("No valid games could be analyzed due to errors.")
        return

    correct_count = sum(valid_results)
    total_valid = len(valid_results)
    accuracy = (correct_count / total_valid) * 100

    print(f"\nAccuracy: {accuracy:.1f}% ({correct_count}/{total_valid} cases solved)")


if __name__ == "__main__":
    asyncio.run(main())
