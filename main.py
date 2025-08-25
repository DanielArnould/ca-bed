"""
Unused for now, but will be an entry point for eval
and API later.
"""

import logging

from method import Method
from models import Model
from tasks.task import InteractionMode
from tasks.twenty_questions.tasks import Bayesian


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    )

    model = Model.DUMMY
    task = Bayesian(max_question_nodes=1, interaction_mode=InteractionMode.BENCHMARK)
    method = Method(model, task, 3, 20, 0.8)
    history = method.run()

    # print(history)


if __name__ == "__main__":
    main()
