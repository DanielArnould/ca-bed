from typing import TypedDict

from history import RunRecord


class RunEval(TypedDict):
    success: bool
    conversation_length: int


class GroupEval(TypedDict):
    success_rate: float
    mean_conversation_length: float
    mean_conversation_length_in_successful_cases: float


def get_run_eval(run_history: RunRecord) -> RunEval:
    did_pass = (
        run_history.true_answer.strip().lower()
        in run_history.final_answer.strip().lower()
    )

    conversation_length = len(run_history.final_path) // 2
    return {"success": did_pass, "conversation_length": conversation_length}


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    success_rate = sum(run_eval["success"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)

    succesful_runs = [run_eval for run_eval in run_evals if run_eval["success"]]
    mean_conversation_length_in_succesful_cases = sum(
        run_eval["conversation_length"] for run_eval in succesful_runs
    ) / len(succesful_runs)

    return {
        "success_rate": success_rate,
        "mean_conversation_length": mean_conversation_length,
        "mean_conversation_length_in_successful_cases": mean_conversation_length_in_succesful_cases,
    }
