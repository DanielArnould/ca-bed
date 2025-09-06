import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from history import serialise_run_history
from method import Method, get_uniform_belief_state
from models import Model
from question_clustering import QuestionClustering
from tasks.med_dg.bayesian import Bayesian
from tasks.med_dg.data import MED_DG_SET, load_data

LOGGER = logging.getLogger("Main")


def main():
    output_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d%H%M%S')}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "setup.log"),
        ],
        force=True,
    )

    benchmark_model = Model.DEEPSEEK_CHAT
    method_model = Model.DEEPSEEK_CHAT
    LOGGER.info(f"Benchmarker: {benchmark_model.name} Method: {method_model.name}")
    question_clustering = QuestionClustering(threshold=0.95)
    initial_belief_state = get_uniform_belief_state(MED_DG_SET)

    method = Method(
        benchmark_model, method_model, question_clustering, initial_belief_state
    )

    dataset = load_data()
    subset = dataset[:50]
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
        for item in subset
    ]

    for i, task in enumerate(tasks):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
            handlers=[
                logging.FileHandler(output_dir / f"{i}.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        history = asyncio.run(method.run(task))
        run_history_path = output_dir / f"{i}.json"

        LOGGER.info(f"Completed run, saving output to {run_history_path}")
        with run_history_path.open("w") as f:
            json.dump(serialise_run_history(history), f)

        LOGGER.info(f"Run saved to {run_history_path}")


if __name__ == "__main__":
    main()
