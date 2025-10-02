import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import direct_prompting_method
from history import (
    save_question_clustering,
    serialise_run_record,
)
import method
from models import LLMRequestSession
from question_clustering import QuestionClustering
from tasks.direct_prompting_task import DirectPromptingTask
from tasks.med_dg.baseline import Baseline
from tasks.med_dg.baseline_multi import BaselineWithMultibranching
from tasks.med_dg.bayesian import Bayesian
from tasks.med_dg.bayesian_multi import BayesianWithMultibranching
from tasks.med_dg.data import MED_DG_SET, load_all_data, load_balanced_data
from tasks.med_dg.direct import Direct
from tasks.task import Task

logger = logging.getLogger("Main")


async def main(output_dir: Path) -> None:
    # =============== CONFIG ===============
    questioner_model_key = "gpt_oss_20b"
    answerer_model_key = "gpt_oss_20b"
    sharpness_constant = 0.4
    min_probability = 0.001
    max_concurrent = 1
    clustering_threshold = 0.97
    dataset = load_balanced_data(0.05)

    tasks = [
        Direct(
            questioner_session=LLMRequestSession(questioner_model_key),
            answerer_session=LLMRequestSession(answerer_model_key),
            task_answer=item.disease,
            # max_question_nodes=2,
            # max_lookahead_depth=1,
            max_conversation_depth=5,
            # confidence_threshold=0.7,
            hypothesis_space=MED_DG_SET,
            self_report=item.self_report,
        )
        for item in dataset[:3]
    ]

    # =============== EXECUTION ===============
    logger.info(f"Questioner: {questioner_model_key} Answerer: {answerer_model_key}")
    question_clustering = QuestionClustering(clustering_threshold)

    semaphore = asyncio.Semaphore(max_concurrent)

    await asyncio.gather(
        *[
            # run_tree_based_task(
            #     i,
            #     task,
            #     output_dir,
            #     semaphore,
            #     sharpness_constant,
            #     min_probability,
            #     question_clustering,
            # )
            run_direct_prompting_task(i, task, output_dir, semaphore)
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
        with run_history_path.open("w", encoding="utf-8") as f:
            json.dump(serialise_run_record(run_record), f)

        logger.info(f"[{idx}] Run saved to {output_dir}")
        logger.info(
            f"[{idx}] Total input tokens: {task.questioner_session.total_input_tokens + task.answerer_session.total_input_tokens}"
            f" Total output tokens: {run_record.questioner_session.total_output_tokens + run_record.answerer_session.total_output_tokens}"
        )


async def run_direct_prompting_task(
    idx: int,
    task: DirectPromptingTask,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        run_record = await direct_prompting_method.run_task(task)
        logger.info(f"[{idx}] Completed run, saving output to {output_dir}")

        run_history_path = output_dir / f"{idx}_run.json"
        with run_history_path.open("w", encoding="utf-8") as f:
            json.dump(serialise_run_record(run_record), f)

        logger.info(f"[{idx}] Run saved to {output_dir}")
        logger.info(
            f"[{idx}] Total input tokens: {task.questioner_session.total_input_tokens + task.answerer_session.total_input_tokens}"
            f" Total output tokens: {run_record.questioner_session.total_output_tokens + run_record.answerer_session.total_output_tokens}"
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
