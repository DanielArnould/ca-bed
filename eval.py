from pathlib import Path
from typing import TypedDict

from history import RunRecord


class RunEval(TypedDict):
    success: bool
    conversation_length: int
    input_tokens: int
    output_tokens: int


class GroupEval(TypedDict):
    success_rate: float
    mean_conversation_length: float
    mean_conversation_length_in_successful_cases: float
    total_input_tokens: int
    total_output_tokens: int


def get_run_eval(run_history: RunRecord) -> RunEval:
    did_pass = (
        run_history.true_answer.strip().lower()
        in run_history.final_answer.strip().lower()
    )

    conversation_length = len(run_history.final_path) // 2
    return {
        "success": did_pass,
        "conversation_length": conversation_length,
        "input_tokens": run_history.total_input_tokens,
        "output_tokens": run_history.total_output_tokens,
    }


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    success_rate = sum(run_eval["success"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)

    succesful_runs = [run_eval for run_eval in run_evals if run_eval["success"]]
    mean_conversation_length_in_succesful_cases = sum(
        run_eval["conversation_length"] for run_eval in succesful_runs
    ) / len(succesful_runs)

    total_input_tokens = sum(run_eval["input_tokens"] for run_eval in run_evals)
    total_output_tokens = sum(run_eval["output_tokens"] for run_eval in run_evals)

    return {
        "success_rate": success_rate,
        "mean_conversation_length": mean_conversation_length,
        "mean_conversation_length_in_successful_cases": mean_conversation_length_in_succesful_cases,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


if __name__ == "__main__":
    import argparse
    from history import deserialise_run_record
    import json
    from tqdm import tqdm

    parser = argparse.ArgumentParser(prog="Experiment evaluator")
    parser.add_argument("-path", "--path", type=Path)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=-1)
    args = parser.parse_args()

    logs_dir: Path = args.path
    start: int = args.start
    end: int = args.end

    run_evals: list[RunEval] = []
    for path in tqdm(logs_dir.rglob("*run.json")):
        with path.open("r") as f:
            run_record = deserialise_run_record(json.load(f))
            run_evals.append(get_run_eval(run_record))

    group_eval = get_group_eval(run_evals[start:end])
    print("=" * 60)
    print(f"Success rate: {group_eval['success_rate']}")
    print(f"Mean conversation length: {group_eval['mean_conversation_length']}")
    print(
        f"Mean conversation length in successful cases: {group_eval['mean_conversation_length_in_successful_cases']}"
    )
    print(f"Total input tokens: {group_eval['total_input_tokens']}")
    print(f"Total output tokens: {group_eval['total_output_tokens']}")

    input_price, output_price = 0.05, 0.2
    input_cost = input_price * (group_eval["total_input_tokens"] / 1_000_000)
    output_cost = output_price * (group_eval["total_output_tokens"] / 1_000_000)

    print(f"Total cost: {input_cost} + {output_cost} = {input_cost + output_cost}")

    print("=" * 60)
