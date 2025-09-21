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
    num_runs: int
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
        "num_runs": len(run_evals),
        "top1": top1,
        "top3": top3,
        "mean_conversation_length": mean_conversation_length,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from history import deserialise_run_record
    import json
    from tqdm import tqdm
    import polars as pl

    parser = argparse.ArgumentParser(prog="Experiment evaluator")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="List of directories to evaluate",
    )
    parser.add_argument(
        "--input-price",
        type=float,
        default=0.05,
        help="Input token price per 1M tokens",
    )
    parser.add_argument(
        "--output-price",
        type=float,
        default=0.2,
        help="Output token price per 1M tokens",
    )
    args = parser.parse_args()

    paths: list[Path] = args.paths
    input_price: float = args.input_price
    output_price: float = args.output_price

    results = []

    for dir_path in paths:
        run_evals: list[RunEval] = []
        for path in tqdm(dir_path.rglob("*run.json"), desc=f"Loading {dir_path.name}"):
            with path.open("r") as f:
                run_record = deserialise_run_record(json.load(f))
            run_evals.append(get_run_eval(run_record))

        group_eval = get_group_eval(run_evals)

        input_cost = input_price * (group_eval["total_input_tokens"] / 1_000_000)
        output_cost = output_price * (group_eval["total_output_tokens"] / 1_000_000)
        total_cost = input_cost + output_cost

        results.append(
            {
                "directory": str(dir_path.name),
                "num_runs": group_eval["num_runs"],
                "top1": group_eval["top1"],
                "top3": group_eval["top3"],
                "mean_conversation_length": group_eval["mean_conversation_length"],
                "total_input_tokens": group_eval["total_input_tokens"],
                "total_output_tokens": group_eval["total_output_tokens"],
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            }
        )

    df = pl.DataFrame(results)

    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 100)

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(df)
