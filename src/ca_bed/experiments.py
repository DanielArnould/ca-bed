import asyncio
from pathlib import Path
import pickle
import random
import time
from typing import Callable, Awaitable

from loguru import logger
from tqdm.asyncio import tqdm
from asyncio import Semaphore

from ca_bed.llm import LLM

from ca_bed.methods.direct_method import run_direct_task
from ca_bed.methods.tree_based_method import run_tree_based_task
from ca_bed.tasks.detective_cases.bayesian import DetectiveCasesBayesian
from ca_bed.tasks.detective_cases.bayesian_multi import (
    DetectiveCasesBayesianMultibranching,
)
from ca_bed.tasks.detective_cases.data import load_all_data
from ca_bed.tasks.detective_cases.direct import DetectiveCasesDirect
from ca_bed.tasks.detective_cases.uot import DetectiveCasesUoT
from ca_bed.tasks.task import Task
from ca_bed.history import RunRecord


async def run_and_save_task(
    task: Task,
    runner_func: Callable[[Task], Awaitable[RunRecord]],
    results_dir: Path,
    method_name: str,
    semaphore: Semaphore,
) -> None:
    async with semaphore:
        task_dir = results_dir / method_name / task.get_id()
        task_dir.mkdir(parents=True, exist_ok=True)

        log_file = task_dir / "log.log"
        handler_id = logger.add(
            log_file,
            filter=lambda record: record["extra"].get("task_id") == task.get_id(),
        )

        with logger.contextualize(task_id=task.get_id()):
            run_record = await runner_func(task)

        run_record_file = task_dir / "run_record.pickle"
        with run_record_file.open("wb") as f:
            pickle.dump(run_record, f)

        logger.remove(handler_id)


async def main() -> None:
    start = time.perf_counter()
    experiment_name = "testy"
    results_dir = Path(f"results/{experiment_name}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    questioner_llm = LLM(model="deepseek-chat")
    answerer_llm = LLM(model="deepseek-chat")

    # --- Hyperparameters ---
    random.seed(42)
    n_questions = 3
    max_conversation_depth = 10
    max_lookahead_depth = 2
    confidence_threshold = 1.0
    estimator_confidence = 1.0
    max_concurrent_tasks = 30
    sample_size = 30

    dataset = load_all_data()
    sample = dataset[10:50]

    # List of tuples: (Method Name, Task Instance, Runner Function)
    tasks_to_run = []

    for case in sample:
        tasks_to_run.append(
            (
                "UoT_D2_0.7_0.8",
                DetectiveCasesUoT(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    case=case,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=max_lookahead_depth,
                    confidence_threshold=0.8,
                    estimator_confidence=0.7,
                ),
                run_tree_based_task,
            )
        )

        tasks_to_run.append(
            (
                "UoT_D2_1.0_1.0",
                DetectiveCasesUoT(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    case=case,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=max_lookahead_depth,
                    confidence_threshold=1.0,
                    estimator_confidence=1.0,
                ),
                run_tree_based_task,
            )
        )

        tasks_to_run.append(
            (
                "UoT_D2_1.0_0.8",
                DetectiveCasesUoT(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    case=case,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=max_lookahead_depth,
                    confidence_threshold=0.8,
                    estimator_confidence=1.0,
                ),
                run_tree_based_task,
            )
        )

        tasks_to_run.append(
            (
                "UoT_D2_0.7_1.0",
                DetectiveCasesUoT(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    case=case,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=max_lookahead_depth,
                    confidence_threshold=1.0,
                    estimator_confidence=0.7,
                ),
                run_tree_based_task,
            )
        )

        tasks_to_run.append(
            (
                "BayesianD2",
                DetectiveCasesBayesian(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    case=case,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=max_lookahead_depth,
                    confidence_threshold=0.8,
                    estimator_confidence=0.7,
                ),
                run_tree_based_task,
            )
        )

    semaphore = Semaphore(max_concurrent_tasks)

    await tqdm.gather(
        *[
            run_and_save_task(
                task=task,
                runner_func=runner_func,
                results_dir=results_dir,
                method_name=method_name,
                semaphore=semaphore,
            )
            for method_name, task, runner_func in tasks_to_run
        ],
    )

    duration = time.perf_counter() - start
    print(f"Completed in {duration}s")


if __name__ == "__main__":
    asyncio.run(main())
