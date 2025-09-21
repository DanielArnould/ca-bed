from pathlib import Path
from typing import TypedDict

from history import RunRecord


class RunEval(TypedDict):
    top1: bool
    top3: bool
    conversation_length: int
    input_tokens: int
    output_tokens: int


class GroupEval(TypedDict):
    top1: float
    top3: float
    mean_conversation_length: float
    total_input_tokens: int
    total_output_tokens: int


def get_run_eval(run_history: RunRecord) -> RunEval:
    guesses = sorted(
        run_history.final_belief_state.keys(),
        key=run_history.final_belief_state.__getitem__,
        reverse=True,
    )
    top3_guesses = guesses[:3]
    top1_guesses = guesses[:1]

    top1 = run_history.true_answer in top1_guesses
    top3 = run_history.true_answer in top3_guesses

    conversation_length = len(run_history.final_path) // 2
    return {
        "top1": top1,
        "top3": top3,
        "conversation_length": conversation_length,
        "input_tokens": run_history.total_input_tokens,
        "output_tokens": run_history.total_output_tokens,
    }


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    top1 = sum(run_eval["top1"] for run_eval in run_evals) / len(run_evals)
    top3 = sum(run_eval["top3"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)

    total_input_tokens = sum(run_eval["input_tokens"] for run_eval in run_evals)
    total_output_tokens = sum(run_eval["output_tokens"] for run_eval in run_evals)

    return {
        "top1": top1,
        "top3": top3,
        "mean_conversation_length": mean_conversation_length,
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
    print(f"Top1: {group_eval['top1']}")
    print(f"Top3: {group_eval['top3']}")
    print(f"Mean conversation length: {group_eval['mean_conversation_length']}")
    print(f"Total input tokens: {group_eval['total_input_tokens']}")
    print(f"Total output tokens: {group_eval['total_output_tokens']}")

    input_price, output_price = 0.05, 0.2
    input_cost = input_price * (group_eval["total_input_tokens"] / 1_000_000)
    output_cost = output_price * (group_eval["total_output_tokens"] / 1_000_000)

    print(f"Total cost: {input_cost} + {output_cost} = {input_cost + output_cost}")

    print("=" * 60)
