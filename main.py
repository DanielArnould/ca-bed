import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from history import (
    save_question_clustering,
    serialise_run_record,
)
import method
from models import LLMRequestSession
from question_clustering import QuestionClustering
from tasks.task import Task
from tasks.twenty_questions.bayesian import Bayesian
from tasks.twenty_questions.data import COMMON

logger = logging.getLogger("Main")


async def main(output_dir: Path) -> None:
    # =============== CONFIG ===============
    questioner_model_key = "gpt_oss_20b"
    answerer_model_key = "gpt_oss_20b"
    sharpness_constant = 0.4
    min_probability = 0.001
    max_concurrent = 1
    clustering_threshold = 0.97
    dataset = COMMON
    random.seed(42)

    tasks = [
        Bayesian(
            questioner_session=LLMRequestSession(questioner_model_key),
            answerer_session=LLMRequestSession(answerer_model_key),
            task_answer=item,
            max_question_nodes=2,
            max_lookahead_depth=3,
            max_conversation_depth=20,
            confidence_threshold=0.7,
            hypothesis_space=dataset,
        )
        for item in random.sample(dataset, 10)
    ]

    # =============== EXECUTION ===============
    logger.info(f"Questioner: {questioner_model_key} Answerer: {answerer_model_key}")
    question_clustering = QuestionClustering(clustering_threshold)

    semaphore = asyncio.Semaphore(max_concurrent)

    await asyncio.gather(
        *[
            run_tree_based_task(
                i,
                task,
                output_dir,
                semaphore,
                sharpness_constant,
                min_probability,
                question_clustering,
            )
            for i, task in enumerate(tasks)
        ]
    )

    logger.info("All runs completed successfully!")


async def run_tree_based_task(
    idx: int,
    task: Task,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    sharpness_constant: float,
    min_probability: float,
    question_clustering: QuestionClustering,
) -> None:
    async with semaphore:
        run_record = await method.run_task(
            task, question_clustering, sharpness_constant, min_probability
        )

        logger.info(f"[{idx}] Completed run, saving output to {output_dir}")
        question_clustering_path_json = output_dir / f"{idx}_cluster.json"
        question_clustering_path_voyager = output_dir / f"{idx}_cluster.voy"

        save_question_clustering(
            question_clustering,
            question_clustering_path_json,
            question_clustering_path_voyager,
        )

        run_history_path = output_dir / f"{idx}_run.json"
        with run_history_path.open("w") as f:
            json.dump(serialise_run_record(run_record), f)

        logger.info(f"[{idx}] Run saved to {output_dir}")
        logger.info(
            f"[{idx}] Total input tokens: {run_record.total_input_tokens} Total output tokens: {run_record.total_output_tokens}"
        )


if __name__ == "__main__":
    output_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d%H%M%S')}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "logs.log", encoding="utf-8"),
        ],
        force=True,
    )
    asyncio.run(main(output_dir))
