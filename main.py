"""
Unused for now, but will be an entry point for eval
and API later.
"""

import asyncio
from datetime import datetime
import json
import multiprocessing
from pathlib import Path

from experiment_logging import serialise_run_history
from loggers import get_logger, setup_logger
from method import Method
from models import Model
from tasks.task import Task
from tasks.twenty_questions.data import COMMON
from tasks.twenty_questions.tasks import Bayesian


def run_task(task: Task, model: Model):
    current_time_formatted = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_process = multiprocessing.current_process()
    current_process.name = f"{task.__class__.__name__}-{task.task_answer}"
    output_dir = Path(f"logs/{current_time_formatted}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("Main", output_dir)
    setup_logger("Method", output_dir)
    setup_logger("LLM Models", output_dir)
    logger = get_logger("Main")

    method = Method(
        model,
        task,
    )
    history = asyncio.run(method.run())
    run_history_path = output_dir / f"{current_process.name}_run.json"

    logger.info(f"Completed run, saving output to {run_history_path}")
    with run_history_path.open("w") as f:
        json.dump(serialise_run_history(history), f)

    logger.info(f"Run saved to {run_history_path}")


def main():
    model = Model.DEEPSEEK_CHAT
    tasks = [
        Bayesian(
            task_answer=item,
            max_question_nodes=3,
            max_lookahead_depth=3,
            max_conversation_depth=20,
            confidence_threshold=0.7,
            hypothesis_space=COMMON,
        )
        for item in COMMON[1:2]
    ]

    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(
        processes=num_cores,
    ) as pool:
        pool.starmap(run_task, [(task, model) for task in tasks])


if __name__ == "__main__":
    main()
