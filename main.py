"""
Unused for now, but will be an entry point for eval
and API later.
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from experiment_logging import serialise_run_history
from method import Method
from models import Model
from tasks.twenty_questions.tasks import Bayesian

LOGGER = logging.getLogger("Main")


def main():
    current_time_formatted = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{current_time_formatted}.log"),
            logging.StreamHandler(),
        ],
    )

    model = Model.DEEPSEEK_CHAT
    task = Bayesian(
        task_answer="Cookie",
        max_question_nodes=2,
        max_lookahead_depth=1,
        max_conversation_depth=2,
        confidence_threshold=0.8,
        hypothesis_space=["Dog", "Cookie", "Paint", "Hat"],
    )
    method = Method(
        model,
        task,
    )
    history = asyncio.run(method.run())
    output_path = Path("logs", f"{current_time_formatted}_run.json")

    LOGGER.info(f"Completed run, saving output to {output_path}")
    with output_path.open("w") as f:
        json.dump(serialise_run_history(history), f)

    LOGGER.info(f"Run saved to {output_path}")


if __name__ == "__main__":
    main()
