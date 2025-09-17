import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from history import (
    deserialise_question_clustering,
    serialise_question_clustering,
    serialise_run_history,
)
from method import Method, get_uniform_belief_state
from models import TOKEN_COUNTER, Model
from question_clustering import QuestionClustering
from tasks.med_dg.bayesian import Bayesian
from tasks.med_dg.data import MED_DG_SET, load_data
from tasks.task import Task

LOGGER = logging.getLogger("Main")


async def main() -> None:
    output_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d%H%M%S')}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "log.log"),
        ],
        force=True,
    )

    benchmark_model = Model.DEEPSEEK_CHAT_TOGETHER_AI
    method_model = Model.DEEPSEEK_CHAT_TOGETHER_AI
    token_counter = TOKEN_COUNTER
    LOGGER.info(f"Benchmarker: {benchmark_model.name} Method: {method_model.name}")
    with open("logs/20250917133204/pre_119_cluster.json", "r") as f:
        question_clustering = deserialise_question_clustering(json.load(f))
    # question_clustering = QuestionClustering()

    dataset = load_data()

    tasks = [
        Bayesian(
            task_answer=item.disease,
            max_question_nodes=3,
            max_lookahead_depth=3,
            max_conversation_depth=5,
            hypothesis_space=MED_DG_SET,
            confidence_threshold=0.7,
            self_report=item.self_report,
        )
        for item in dataset[119:247]
    ]

    max_concurrent = 1
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_sample_with_semaphore(idx: int, task: Task) -> None:
        async with semaphore:
            question_clustering_path = output_dir / f"pre_{idx:03d}_cluster.json"
            run_history_path = output_dir / f"{idx:03d}_run.json"

            with question_clustering_path.open("w") as f:
                json.dump(serialise_question_clustering(question_clustering), f)

            method = Method(
                benchmark_model,
                method_model,
                question_clustering,
                get_uniform_belief_state(MED_DG_SET),
            )
            history = await method.run(task)

            LOGGER.info(f"Completed run, saving output to {run_history_path}")
            with run_history_path.open("w") as f:
                json.dump(serialise_run_history(history), f)

            LOGGER.info(f"Run saved to {run_history_path}")
            LOGGER.info(
                f"{token_counter.total_input_tokens=} {token_counter.total_output_tokens=}"
            )

    await asyncio.gather(
        *[run_sample_with_semaphore(i, task) for i, task in enumerate(tasks, start=119)]
    )

    with (output_dir / "final_cluster.json").open("w") as f:
        json.dump(serialise_question_clustering(question_clustering), f)


if __name__ == "__main__":
    asyncio.run(main())
