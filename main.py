"""
Unused for now, but will be an entry point for eval
and API later.
"""

import json
import logging
from pathlib import Path

from experiment_logging import serialise_run_history
from method import Method
from models import Model
from tasks.task import InteractionMode
from tasks.twenty_questions.tasks import Bayesian

LOGGER = logging.getLogger("Main")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    )

    model = Model.DUMMY
    task = Bayesian(max_question_nodes=2, interaction_mode=InteractionMode.BENCHMARK)
    method = Method(
        model,
        task,
        max_lookahead_depth=1,
        max_conversation_depth=1,
        confidence_threshold=0.8,
    )
    history = method.run()
    output_path = Path("run.json")

    LOGGER.info("Completed run, saving output to run.json...")
    with output_path.open("w") as f:
        json.dump(serialise_run_history(history), f)

    LOGGER.info("Run saved to run.json")


if __name__ == "__main__":
    main()
