import asyncio
from pathlib import Path
import pickle
import random
import sys

from loguru import logger
from tqdm.asyncio import tqdm

from ca_bed.llm import LLM
from ca_bed.method import run_task
from ca_bed.tasks.twenty_questions.bayesian import TwentyQuestionsBayesian
from ca_bed.tasks.twenty_questions.data import TWENTY_QUESTIONS_ENTITIES
from ca_bed.tasks.task import Task

from asyncio import Semaphore

from ca_bed.tasks.twenty_questions.uot import TwentyQuestionsUoT


async def run_and_save_task(
    task: Task,
    results_dir: Path,
    questioner_llm: LLM,
    answerer_llm: LLM,
    n_questions: int,
    max_conversation_depth: int,
    max_lookahead_depth: int,
    confidence_threshold: float,
    semaphore: Semaphore,
) -> None:
    async with semaphore:
        log_file = results_dir / task.get_id() / "log.log"
        logger.add(
            log_file,
            filter=lambda record: record["extra"].get("task_id") == task.get_id(),
        )

        run_record = await run_task(
            task=task,
            questioner_llm=questioner_llm,
            answerer_llm=answerer_llm,
            n_questions=n_questions,
            max_conversation_depth=max_conversation_depth,
            max_lookahead_depth=max_lookahead_depth,
            confidence_threshold=confidence_threshold,
        )

        run_record_file = results_dir / task.get_id() / "root.pickle"
        with run_record_file.open("wb") as f:
            pickle.dump(run_record, f)


async def main() -> None:
    results_dir = Path("results4/")
    results_dir.mkdir(parents=True, exist_ok=False)
    logger.remove(0)

    global_log_file = results_dir / "log.log"
    logger.add(global_log_file, filter=lambda record: "task_id" not in record["extra"])
    logger.add(sys.stdout, filter=lambda record: "task_id" not in record["extra"])

    questioner_llm = LLM(model="deepseek-chat")
    answerer_llm = LLM(model="deepseek-chat")

    logger.info(f"Questioner LLM: {questioner_llm.model}")
    logger.info(f"Answerer LLM: {answerer_llm.model}")

    random.seed(42)
    n_questions = 3
    max_conversation_depth = 10
    max_lookahead_depth = 2
    confidence_threshold = 0.9
    max_concurrent_tasks = 8

    secret_entities = random.sample(
        TWENTY_QUESTIONS_ENTITIES, len(TWENTY_QUESTIONS_ENTITIES)
    )

    tasks: list[Task] = [
        TwentyQuestionsUoT(
            secret_entity=secret_entity,
            entities=secret_entities,
        )
        for secret_entity in [secret_entities[0]]
    ]

    semaphore = Semaphore(max_concurrent_tasks)

    await tqdm.gather(
        *[
            run_and_save_task(
                task=task,
                results_dir=results_dir,
                questioner_llm=questioner_llm,
                answerer_llm=answerer_llm,
                n_questions=n_questions,
                max_conversation_depth=max_conversation_depth,
                max_lookahead_depth=max_lookahead_depth,
                confidence_threshold=confidence_threshold,
                semaphore=semaphore,
            )
            for task in tasks
        ],
        desc="Running tasks",
        total=len(tasks),
    )

    logger.info(f"Total questioner input tokens: {questioner_llm.total_input_tokens}")
    logger.info(f"Total questioner output tokens: {questioner_llm.total_output_tokens}")
    logger.info(f"Total answerer input tokens: {answerer_llm.total_input_tokens}")
    logger.info(f"Total answerer output tokens: {answerer_llm.total_output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
