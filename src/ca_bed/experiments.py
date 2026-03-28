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

from ca_bed.methods.tree_based_method import run_tree_based_task
from ca_bed.tasks.task import Task
from ca_bed.history import RunRecord
from ca_bed.tasks.twenty_questions.bayesian import TwentyQuestionsBayesian
from ca_bed.tasks.twenty_questions.data import TWENTY_QUESTIONS_ENTITIES


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
    experiment_name = "scaling_ablation_twenty_questions"
    results_dir = Path(f"results/{experiment_name}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    questioner_llm = LLM(model="deepseek-chat")
    answerer_llm = LLM(model="deepseek-chat")

    # --- Hyperparameters ---
    random.seed(42)
    max_conversation_depth = 20
    max_concurrent_tasks = 40
    sample_size = 20

    confidence_threshold = 0.8
    estimator_confidence = 0.7

    dataset = TWENTY_QUESTIONS_ENTITIES
    sampled_dataset = random.sample(dataset, len(dataset))[:sample_size]

    tasks_to_run = []

    # --- 1D Sweep Configurations ---
    # Format: (Width, Depth, Method Label)
    ablation_configs = [
        # 2. The Depth Sweep (Holding Width at 3)
        (3, 1, "DepthSweep_W3_D1"),
        (3, 2, "Baseline_W3_D2"),
        (3, 3, "DepthSweep_W3_D3"),
        (3, 3, "DepthSweep_W3_D4"),
        # 3. The Width Sweep (Holding Depth at 2)
        (2, 2, "WidthSweep_W2_D2"),
        (5, 2, "WidthSweep_W5_D2"),
    ]

    for width, depth, method_name in ablation_configs:
        for entity in sampled_dataset:
            tasks_to_run.append(
                (
                    method_name,
                    TwentyQuestionsBayesian(
                        questioner_llm=questioner_llm,
                        answerer_llm=answerer_llm,
                        task_answer=entity,
                        hypothesis_space=dataset,
                        max_conversation_depth=max_conversation_depth,
                        n_questions=width,
                        max_lookahead_depth=depth,
                        confidence_threshold=confidence_threshold,
                        estimator_confidence=estimator_confidence,
                    ),
                    run_tree_based_task,
                )
            )

    print(f"Total tasks queued: {len(tasks_to_run)}")
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
    print(f"Completed in {duration:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
