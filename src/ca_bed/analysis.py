import argparse
from pathlib import Path
import pickle
from typing import TypedDict

from tqdm import tqdm

from ca_bed.history import RunRecord


class RunEval(TypedDict):
    top1: bool
    top3: bool
    conversation_length: int


class GroupEval(TypedDict):
    num_runs: int
    top1: float
    top3: float
    mean_conversation_length: float
    mean_conversation_length_in_successful_cases: float


def get_run_eval(run_record: RunRecord) -> RunEval:
    conversation_length = len(run_record.final_path) - 1
    final_belief_state = run_record.final_path[-1].belief_state

    guesses = sorted(
        final_belief_state.keys(),
        key=final_belief_state.__getitem__,
        reverse=True,
    )
    top3_guesses = guesses[:3]
    top1_guesses = guesses[:1]

    top1 = run_record.task.get_expected_answer() in top1_guesses
    top3 = run_record.task.get_expected_answer() in top3_guesses

    return {"top1": top1, "top3": top3, "conversation_length": conversation_length}


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    top1 = sum(run_eval["top1"] for run_eval in run_evals) / len(run_evals)
    top3 = sum(run_eval["top3"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)
    conv_length_successful = [
        run_eval["conversation_length"] for run_eval in run_evals if run_eval["top1"]
    ]
    mean_conversation_length_in_successful_cases = sum(conv_length_successful) / len(
        conv_length_successful
    )

    return {
        "num_runs": len(run_evals),
        "top1": top1,
        "top3": top3,
        "mean_conversation_length": mean_conversation_length,
        "mean_conversation_length_in_successful_cases": mean_conversation_length_in_successful_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="Experiment Analysis")
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Directory to evaluate",
    )
    args = parser.parse_args()

    path: Path = args.path

    run_evals: list[RunEval] = []

    for path in tqdm(path.rglob("*.pickle")):
        with path.open("rb") as f:
            run_record = pickle.load(f)

        run_evals.append(get_run_eval(run_record))

    group_eval = get_group_eval(run_evals)
    print(group_eval)


if __name__ == "__main__":
    main()
