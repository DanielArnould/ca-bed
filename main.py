import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path

from direct_prompting_method import DirectPromptingMethod
from history import (
    load_question_clustering,
    save_question_clustering,
    serialise_run_record,
)
from method import Method
from models import Model
from question_clustering import QuestionClustering
from tasks.direct_prompting_task import DirectPromptingTask
from tasks.twenty_questions.bayesian import Bayesian
from tasks.twenty_questions.baseline import Baseline
from tasks.movie_lens.baseline import Baseline as MovieLensBaseline
from tasks.movie_lens.bayesian import Bayesian as MovieLensBayesian
from tasks.movie_lens.data import load_balanced_dataset
from tasks.movie_lens.common import persona_record_to_context
from tasks.task import Task
from tasks.twenty_questions.data import (
    BIG_BENCH_CONCEPT,
    Animals,
    Food,
    Objects,
    Places,
)

LOGGER = logging.getLogger("Main")


def setup_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "logs.log", encoding="utf-8"),
        ],
        force=True,
    )


async def run_single_direct_prompting_task(
    idx: int,
    task: DirectPromptingTask,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    benchmark_model: Model,
    method_model: Model,
) -> None:
    async with semaphore:
        method = DirectPromptingMethod(
            benchmark_model,
            method_model,
            task,
        )
        history = await method.run()

        LOGGER.info(f"[{idx}] Completed run, saving output to {output_dir}")

        run_history_path = output_dir / f"{idx}_run.json"
        with run_history_path.open("w") as f:
            json.dump(serialise_run_record(history), f)

        LOGGER.info(f"[{idx}] Run saved to {output_dir}")
        LOGGER.info(
            f"[{idx}] Total input tokens: {history.total_input_tokens} Total output tokens: {history.total_output_tokens}"
        )


async def run_single_task(
    idx: int,
    task: Task,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    benchmark_model: Model,
    method_model: Model,
    sharpness_constant: float,
    question_clustering: QuestionClustering,
) -> None:
    async with semaphore:
        method = Method(
            benchmark_model, method_model, sharpness_constant, task, question_clustering
        )
        history = await method.run()

        LOGGER.info(f"[{idx}] Completed run, saving output to {output_dir}")
        question_clustering_path_json = output_dir / f"{idx}_cluster.json"
        question_clustering_path_voyager = output_dir / f"{idx}_cluster.voy"

        save_question_clustering(
            question_clustering,
            question_clustering_path_json,
            question_clustering_path_voyager,
        )

        run_history_path = output_dir / f"{idx}_run.json"
        with run_history_path.open("w") as f:
            json.dump(serialise_run_record(history), f)

        LOGGER.info(f"[{idx}] Run saved to {output_dir}")
        LOGGER.info(
            f"[{idx}] Total input tokens: {history.total_input_tokens} Total output tokens: {history.total_output_tokens}"
        )


async def main() -> None:
    # =============== CONFIG ===============
    benchmark_model = Model.GPT_OSS_20B
    method_model = Model.DEEPSEEK_CHAT_TOGETHER_AI
    sharpness_constant = 0.4
    max_concurrent = 1
    clustering_threshold = 0.97
    dataset, movie_set = load_balanced_dataset()

    tasks = [
        MovieLensBayesian(
            max_question_nodes=2,
            max_conversation_depth=5,
            max_lookahead_depth=3,
            persona=persona_record_to_context(
                item.persona, item.preferred_movies, item.disliked_movies),
            candidate_movies=movie_set,
            confidence_threshold=1
        )
        for item in dataset[0:1]
    ]

    # =============== EXECUTION ===============
    output_dir = Path(f"logs/{datetime.now().strftime('%Y%m%d%H%M%S')}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    LOGGER.info(
        f"Benchmarker: {benchmark_model.name} Method: {method_model.name}")
    question_clustering = QuestionClustering(clustering_threshold)

    semaphore = asyncio.Semaphore(max_concurrent)

    await asyncio.gather(
        *[
            run_single_task(
                i,
                task,
                output_dir,
                semaphore,
                benchmark_model=benchmark_model,
                method_model=method_model,
                sharpness_constant=sharpness_constant,
                question_clustering=question_clustering,
            )
            for i, task in enumerate(tasks)
        ]
    )

    save_question_clustering(
        question_clustering,
        output_dir / "final_cluster.json",
        output_dir / "final_cluster.voy",
    )
    LOGGER.info("All runs completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
    # dataset, movie_set = load_balanced_dataset()
    # item = dataset[0]
    # task = MovieLensBaseline(
    #     max_question_nodes=2,
    #     max_conversation_depth=5,
    #     max_lookahead_depth=3,
    #     persona=persona_record_to_context(
    #         item.persona, item.preferred_movies, item.disliked_movies),
    #     candidate_movies=movie_set
    # )

    # # print(len(movie_set))

    # print(task._persona_block)
