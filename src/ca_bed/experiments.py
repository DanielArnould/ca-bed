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
from ca_bed.tasks.task import Task
from ca_bed.tasks.twenty_questions.bayesian import TwentyQuestionsBayesian
from ca_bed.tasks.twenty_questions.bayesian_multibranching import (
    TwentyQuestionsBayesianMultibranching,
)
from ca_bed.tasks.twenty_questions.uot import TwentyQuestionsUoT
from ca_bed.tasks.twenty_questions.direct import TwentyQuestionsDirect
from ca_bed.tasks.twenty_questions.data import TWENTY_QUESTIONS_ENTITIES
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
    experiment_name = "test5"
    results_dir = Path(f"results/{experiment_name}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    questioner_llm = LLM(model="deepseek-chat")
    answerer_llm = LLM(model="deepseek-chat")

    # --- Hyperparameters ---
    random.seed(42)
    n_questions = 3
    max_conversation_depth = 20
    # max_lookahead_depth = 1
    confidence_threshold = 0.9
    estimator_confidence = 0.9  # IGNORED
    max_concurrent_tasks = 30
    sample_size = 1

    hypothesis_space = TWENTY_QUESTIONS_ENTITIES
    evaluation_set = random.sample(hypothesis_space, len(hypothesis_space))[
        :sample_size
    ]

    # List of tuples: (Method Name, Task Instance, Runner Function)
    tasks_to_run = []

    for secret_entity in evaluation_set:
        # tasks_to_run.append(
        #     (
        #         "BayesianD1",
        #         TwentyQuestionsBayesian(
        #             questioner_llm=questioner_llm,
        #             answerer_llm=answerer_llm,
        #             task_answer=secret_entity,
        #             hypothesis_space=hypothesis_space,
        #             max_conversation_depth=max_conversation_depth,
        #             n_questions=n_questions,
        #             max_lookahead_depth=1,
        #             confidence_threshold=confidence_threshold,
        #             estimator_confidence=estimator_confidence,
        #         ),
        #         run_tree_based_task,
        #     )
        # )

        tasks_to_run.append(
            (
                "BayesianD2",
                TwentyQuestionsBayesian(
                    questioner_llm=questioner_llm,
                    answerer_llm=answerer_llm,
                    task_answer=secret_entity,
                    hypothesis_space=hypothesis_space,
                    max_conversation_depth=max_conversation_depth,
                    n_questions=n_questions,
                    max_lookahead_depth=2,
                    confidence_threshold=confidence_threshold,
                    estimator_confidence=estimator_confidence,
                ),
                run_tree_based_task,
            )
        )

        # tasks_to_run.append(
        #     (
        #         "BayesianD3",
        #         TwentyQuestionsBayesian(
        #             questioner_llm=questioner_llm,
        #             answerer_llm=answerer_llm,
        #             task_answer=secret_entity,
        #             hypothesis_space=hypothesis_space,
        #             max_conversation_depth=max_conversation_depth,
        #             n_questions=n_questions,
        #             max_lookahead_depth=3,
        #             confidence_threshold=confidence_threshold,
        #             estimator_confidence=estimator_confidence,
        #         ),
        #         run_tree_based_task,
        #     )
        # )

        # # 2. Bayesian Multibranching (Multiple Choice)
        # tasks_to_run.append(
        #     (
        #         "BayesianMulti",
        #         TwentyQuestionsBayesianMultibranching(
        #             questioner_llm=questioner_llm,
        #             answerer_llm=answerer_llm,
        #             task_answer=secret_entity,
        #             hypothesis_space=hypothesis_space,
        #             max_conversation_depth=max_conversation_depth,
        #             n_questions=n_questions,
        #             max_lookahead_depth=max_lookahead_depth,
        #             confidence_threshold=confidence_threshold,
        #             estimator_confidence=estimator_confidence,
        #         ),
        #         run_tree_based_task,
        #     )
        # )

        # # 3. Uncertainty of Thought (UoT)
        # tasks_to_run.append(
        #     (
        #         "UoT",
        #         TwentyQuestionsUoT(
        #             questioner_llm=questioner_llm,
        #             answerer_llm=answerer_llm,
        #             task_answer=secret_entity,
        #             hypothesis_space=hypothesis_space,
        #             max_conversation_depth=max_conversation_depth,
        #             n_questions=n_questions,
        #             max_lookahead_depth=max_lookahead_depth,
        #             confidence_threshold=confidence_threshold,
        #             estimator_confidence=estimator_confidence,
        #         ),
        #         run_tree_based_task,
        #     )
        # )

        # 4. Direct Prompting (Linear)
        # tasks_to_run.append(
        #     (
        #         "Direct",
        #         TwentyQuestionsDirect(
        #             questioner_llm=questioner_llm,
        #             answerer_llm=answerer_llm,
        #             task_answer=secret_entity,
        #             hypothesis_space=hypothesis_space,
        #             max_conversation_depth=max_conversation_depth,
        #         ),
        #         run_direct_task,
        #     )
        # )

    semaphore = Semaphore(max_concurrent_tasks)

    # Execute all tasks concurrently
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
        desc=f"Running {len(tasks_to_run)} total tasks",
    )

    duration = time.perf_counter() - start
    print(f"Completed in {duration}s")


if __name__ == "__main__":
    asyncio.run(main())
