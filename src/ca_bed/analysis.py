import argparse
import math
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
    top1_ci_lower: float
    top1_ci_upper: float
    top3: float
    top3_ci_lower: float
    top3_ci_upper: float
    mean_conversation_length: float
    conversation_length_sem: float
    mean_conversation_length_in_successful_cases: float
    conversation_length_success_sem: float


def wilson_score_interval(
    successes: int, n: int, z: float = 1.96
) -> tuple[float, float]:
    """Calculates the Wilson Score Interval for binomial proportions (95% CI by default)."""
    if n == 0:
        return 0.0, 0.0

    p_hat = successes / n
    denominator = 1 + z**2 / n
    center_adjusted_prob = p_hat + z**2 / (2 * n)
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))

    lower_bound = (center_adjusted_prob - z * adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + z * adjusted_std_dev) / denominator

    return max(0.0, lower_bound), min(1.0, upper_bound)


def calculate_sem(values: list[float | int]) -> float:
    """Calculates the Standard Error of the Mean (SEM)."""
    n = len(values)
    if n <= 1:
        return 0.0

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)  # Sample variance
    return math.sqrt(variance / n)


def get_run_eval(run_record: RunRecord) -> RunEval:
    # Safely handle the direct task format vs tree format
    conversation_length = len(run_record.final_path) - 1

    # In direct prompting, belief state is updated incrementally. In Bayesian, it's evaluated.
    final_belief_state = run_record.final_path[-1].belief_state

    guesses = sorted(
        final_belief_state.keys(),
        key=final_belief_state.__getitem__,
        reverse=True,
    )
    top3_guesses = guesses[:3]
    top1_guesses = guesses[:1]

    # Task instance interface unification
    expected_answer = run_record.task.get_task_answer()

    top1 = expected_answer in top1_guesses
    top3 = expected_answer in top3_guesses

    return {"top1": top1, "top3": top3, "conversation_length": conversation_length}


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    n = len(run_evals)
    if n == 0:
        raise ValueError("Cannot evaluate an empty list of runs.")

    top1_successes = sum(1 for r in run_evals if r["top1"])
    top3_successes = sum(1 for r in run_evals if r["top3"])

    top1_rate = top1_successes / n
    top3_rate = top3_successes / n

    top1_ci = wilson_score_interval(top1_successes, n)
    top3_ci = wilson_score_interval(top3_successes, n)

    all_lengths = [r["conversation_length"] for r in run_evals]
    successful_lengths = [r["conversation_length"] for r in run_evals if r["top1"]]

    mean_len = sum(all_lengths) / n
    sem_len = calculate_sem(all_lengths)

    mean_succ_len = (
        sum(successful_lengths) / len(successful_lengths) if successful_lengths else 0.0
    )
    sem_succ_len = calculate_sem(successful_lengths)

    return {
        "num_runs": n,
        "top1": top1_rate,
        "top1_ci_lower": top1_ci[0],
        "top1_ci_upper": top1_ci[1],
        "top3": top3_rate,
        "top3_ci_lower": top3_ci[0],
        "top3_ci_upper": top3_ci[1],
        "mean_conversation_length": mean_len,
        "conversation_length_sem": sem_len,
        "mean_conversation_length_in_successful_cases": mean_succ_len,
        "conversation_length_success_sem": sem_succ_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="Experiment Analysis")
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="Base experiment directory containing method subdirectories",
    )
    args = parser.parse_args()
    base_path: Path = args.path

    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.")
        return

    # Find all immediate subdirectories (these are the method names)
    method_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    results_by_method: dict[str, GroupEval] = {}

    for method_dir in method_dirs:
        run_evals: list[RunEval] = []
        pickle_files = list(method_dir.rglob("*.pickle"))

        if not pickle_files:
            continue

        for path in tqdm(pickle_files, desc=f"Evaluating {method_dir.name}"):
            with path.open("rb") as f:
                run_record = pickle.load(f)
            run_evals.append(get_run_eval(run_record))

        results_by_method[method_dir.name] = get_group_eval(run_evals)

    # --- Print Formatted Results ---
    print("\n" + "=" * 95)
    print(
        f"{'Method':<18} | {'N':<4} | {'Top-1 Acc (95% CI)':<22} | {'Top-3 Acc (95% CI)':<22} | {'Avg Depth (SEM)':<18}"
    )
    print("-" * 95)

    for method, eval_data in sorted(results_by_method.items()):
        # Format Top-1
        t1 = eval_data["top1"] * 100
        t1_l = eval_data["top1_ci_lower"] * 100
        t1_u = eval_data["top1_ci_upper"] * 100
        t1_str = f"{t1:5.1f}% [{t1_l:4.1f}, {t1_u:4.1f}]"

        # Format Top-3
        t3 = eval_data["top3"] * 100
        t3_l = eval_data["top3_ci_lower"] * 100
        t3_u = eval_data["top3_ci_upper"] * 100
        t3_str = f"{t3:5.1f}% [{t3_l:4.1f}, {t3_u:4.1f}]"

        # Format Depth
        depth = eval_data["mean_conversation_length"]
        depth_sem = eval_data["conversation_length_sem"]
        depth_str = f"{depth:4.1f} (±{depth_sem:.2f})"

        print(
            f"{method:<18} | {eval_data['num_runs']:<4} | {t1_str:<22} | {t3_str:<22} | {depth_str:<18}"
        )

    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
