import argparse
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
from ca_bed.history import RunRecord
from ca_bed.tasks.task import Task

from ca_bed.methods.tree_based_method import run_tree_based_task
from ca_bed.methods.direct_method import run_direct_task

from ca_bed.tasks.detective_cases.data import load_all_data as load_detective_data
from ca_bed.tasks.detective_cases.uot import DetectiveCasesUoT
from ca_bed.tasks.detective_cases.bayesian import DetectiveCasesBayesian
from ca_bed.tasks.detective_cases.bayesian_multi import DetectiveCasesBayesianMultibranching
from ca_bed.tasks.detective_cases.direct import DetectiveCasesDirect

from ca_bed.tasks.twenty_questions.data import TWENTY_QUESTIONS_ENTITIES
from ca_bed.tasks.twenty_questions.uot import TwentyQuestionsUoT
from ca_bed.tasks.twenty_questions.bayesian import TwentyQuestionsBayesian
from ca_bed.tasks.twenty_questions.bayesian_multibranching import TwentyQuestionsBayesianMultibranching
from ca_bed.tasks.twenty_questions.direct import TwentyQuestionsDirect


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CA-BED Experiments")

    # Experiment Info
    parser.add_argument("--experiment_name", type=str, default=f"run_{int(time.time())}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, required=True, choices=[
        "detective_direct", "detective_uot", "detective_bayesian", "detective_bayesian_multi",
        "twentyq_direct", "twentyq_uot", "twentyq_bayesian", "twentyq_bayesian_multi"
    ])

    # Common Settings
    parser.add_argument("--questioner_model", default="deepseek-chat")
    parser.add_argument("--answerer_model", default="deepseek-reasoner")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--max_conversation_depth", type=int, default=20)
    parser.add_argument("--max_concurrent_tasks", type=int, default=6)

    # Tree Method Settings
    parser.add_argument("--max_question_nodes", type=int, default=3)
    parser.add_argument("--max_lookahead_depth", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--estimator_confidence", type=float, default=0.7)

    return parser.parse_args()


def create_task_instance(task_name: str, item, args, questioner_llm: LLM, answerer_llm: LLM, dataset) -> Task:
    # Detective Cases
    if task_name == "detective_direct":
        return DetectiveCasesDirect(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            instance=item,
            max_conversation_depth=args.max_conversation_depth,
        )
    elif task_name == "detective_uot":
        return DetectiveCasesUoT(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            instance=item,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )
    elif task_name == "detective_bayesian":
        return DetectiveCasesBayesian(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            instance=item,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )
    elif task_name == "detective_bayesian_multi":
        return DetectiveCasesBayesianMultibranching(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            instance=item,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )

    # Twenty Questions
    elif task_name == "twentyq_direct":
        return TwentyQuestionsDirect(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=item,
            hypothesis_space=dataset,
            max_conversation_depth=args.max_conversation_depth,
        )
    elif task_name == "twentyq_uot":
        return TwentyQuestionsUoT(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=item,
            hypothesis_space=dataset,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )
    elif task_name == "twentyq_bayesian":
        return TwentyQuestionsBayesian(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=item,
            hypothesis_space=dataset,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )
    elif task_name == "twentyq_bayesian_multi":
        return TwentyQuestionsBayesianMultibranching(
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            task_answer=item,
            hypothesis_space=dataset,
            n_questions=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.max_conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


async def main(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    results_dir = Path(f"results/{args.experiment_name}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    questioner_llm = LLM(model=args.questioner_model)
    answerer_llm = LLM(model=args.answerer_model)

    random.seed(args.seed)

    if args.task.startswith("detective_"):
        dataset = load_detective_data()
    else:
        dataset = TWENTY_QUESTIONS_ENTITIES

    sampled_dataset = dataset[args.start_idx : args.end_idx]

    tasks_to_run = []
    method_name = args.task

    # Queue generation
    for item in sampled_dataset:
        task_instance = create_task_instance(
            task_name=method_name,
            item=item,
            args=args,
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            dataset=dataset
        )

        # Delegate execution method based on task type
        runner_func = run_direct_task if "direct" in method_name else run_tree_based_task
        tasks_to_run.append((method_name, task_instance, runner_func))

    print(f"Total tasks queued: {len(tasks_to_run)}")
    semaphore = Semaphore(args.max_concurrent_tasks)

    # Parallel Execution 
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
    args = parse_args()
    asyncio.run(main(args))