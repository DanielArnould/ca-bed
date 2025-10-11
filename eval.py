from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Sequence, TypedDict

from history import RunRecord
from models import LLMRequestSession, query_llm
from tasks.movie_lens.common import PersonaContext, format_persona_context

import json


class RunEval(TypedDict):
    top1: bool
    top3: bool
    conversation_length: int
    questioner_input_tokens: int
    questioner_output_tokens: int
    answerer_input_tokens: int
    answerer_output_tokens: int


class GroupEval(TypedDict):
    num_runs: int
    top1: float
    top3: float
    mean_conversation_length: float
    questioner_input_tokens: int
    questioner_output_tokens: int
    answerer_input_tokens: int
    answerer_output_tokens: int


class MovieLensEval(TypedDict):
    recommendations: tuple[str, ...]
    ratings: dict[str, float]
    mean_rating: float | None
    missing_titles: tuple[str, ...]
    raw_response: str
    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    eval_session: LLMRequestSession


class MovieLensGroupEval(TypedDict):
    num_runs: int
    expected_ratings: int
    captured_ratings: int
    rating_coverage: float
    overall_mean_rating: float | None
    mean_mean_rating: float | None
    missing_titles: dict[str, int]
    questioner_input_tokens: int
    questioner_output_tokens: int
    answerer_input_tokens: int
    answerer_output_tokens: int
    eval_input_tokens: int
    eval_output_tokens: int


def get_run_eval(run_history: RunRecord) -> RunEval:
    guesses = sorted(
        run_history.final_belief_state.keys(),
        key=run_history.final_belief_state.__getitem__,
        reverse=True,
    )
    top3_guesses = guesses[:3]
    top1_guesses = guesses[:1]

    top1 = run_history.expected_answer in top1_guesses
    top3 = run_history.expected_answer in top3_guesses

    conversation_length = len(run_history.final_path) // 2
    return {
        "top1": top1,
        "top3": top3,
        "conversation_length": conversation_length,
        "questioner_input_tokens": run_history.questioner_session.total_input_tokens,
        "questioner_output_tokens": run_history.questioner_session.total_output_tokens,
        "answerer_input_tokens": run_history.answerer_session.total_input_tokens,
        "answerer_output_tokens": run_history.answerer_session.total_output_tokens,
    }


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    top1 = sum(run_eval["top1"] for run_eval in run_evals) / len(run_evals)
    top3 = sum(run_eval["top3"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)

    questioner_input_tokens = sum(
        run_eval["questioner_input_tokens"] for run_eval in run_evals
    )
    questioner_output_tokens = sum(
        run_eval["questioner_output_tokens"] for run_eval in run_evals
    )
    answerer_input_tokens = sum(
        run_eval["answerer_input_tokens"] for run_eval in run_evals
    )
    answerer_output_tokens = sum(
        run_eval["answerer_output_tokens"] for run_eval in run_evals
    )

    return {
        "num_runs": len(run_evals),
        "top1": top1,
        "top3": top3,
        "mean_conversation_length": mean_conversation_length,
        "questioner_input_tokens": questioner_input_tokens,
        "questioner_output_tokens": questioner_output_tokens,
        "answerer_input_tokens": answerer_input_tokens,
        "answerer_output_tokens": answerer_output_tokens,
    }


def _select_top_titles(belief_state: dict[str, float], limit: int) -> tuple[str, ...]:
    ordered = sorted(
        belief_state.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(title for title, _score in ordered[:limit])


def _parse_rating_response(
    response: str, expected_titles: Sequence[str]
) -> dict[str, float]:
    return json.loads(response)


def get_movielens_group_eval(
    movie_evals: list[MovieLensEval],
) -> MovieLensGroupEval:
    total_expected = sum(
        len(movie_eval["recommendations"]) for movie_eval in movie_evals
    )
    total_captured = sum(len(movie_eval["ratings"]) for movie_eval in movie_evals)

    total_questioner_input_tokens = sum(
        movie_eval["questioner_session"].total_input_tokens
        for movie_eval in movie_evals
    )
    total_questioner_output_tokens = sum(
        movie_eval["questioner_session"].total_output_tokens
        for movie_eval in movie_evals
    )
    total_answerer_input_tokens = sum(
        movie_eval["answerer_session"].total_input_tokens for movie_eval in movie_evals
    )
    total_answerer_output_tokens = sum(
        movie_eval["answerer_session"].total_output_tokens for movie_eval in movie_evals
    )
    total_eval_input_tokens = sum(
        movie_eval["eval_session"].total_input_tokens for movie_eval in movie_evals
    )
    total_eval_output_tokens = sum(
        movie_eval["eval_session"].total_output_tokens for movie_eval in movie_evals
    )

    all_scores: list[float] = []
    mean_scores: list[float] = []
    missing_counter: Counter[str] = Counter()

    for movie_eval in movie_evals:
        all_scores.extend(movie_eval["ratings"].values())
        if movie_eval["mean_rating"] is not None:
            mean_scores.append(movie_eval["mean_rating"])
        missing_counter.update(movie_eval["missing_titles"])

    overall_mean = round(sum(all_scores) / len(all_scores), 3) if all_scores else None
    mean_of_means = (
        round(sum(mean_scores) / len(mean_scores), 3) if mean_scores else None
    )

    return {
        "num_runs": len(movie_evals),
        "expected_ratings": total_expected,
        "captured_ratings": total_captured,
        "overall_mean_rating": overall_mean,
        "mean_mean_rating": mean_of_means,
        "questioner_input_tokens": total_questioner_input_tokens,
        "questioner_output_tokens": total_questioner_output_tokens,
        "answerer_input_tokens": total_answerer_input_tokens,
        "answerer_output_tokens": total_answerer_output_tokens,
        "eval_input_tokens": total_eval_input_tokens,
        "eval_output_tokens": total_eval_output_tokens,
    }


async def get_movielens_eval(
    run_history: RunRecord,
    persona: PersonaContext,
    evaluation_session: LLMRequestSession,
    max_recommendations: int = 10,
) -> MovieLensEval:
    top_titles = _select_top_titles(run_history.final_belief_state, max_recommendations)

    persona_block = format_persona_context(persona)
    movies_block = "\n".join(
        f"{idx}. {title}" for idx, title in enumerate(top_titles, start=1)
    )

    prompt = dedent(
        f"""\
        You are evaluating how well a recommendation list matches a specific movie-goer persona so a reviewer can see which titles truly fit.

        ### Persona Briefing
        {persona_block}

        ### Candidate Films
        {movies_block}

        Consider the persona's stated likes, dislikes, motifs, and reference titles. Reward close matches, penalise strong aversions, and prefer variety over near-duplicates. If you lack enough information for a film, give the best informed estimate.

        ### Rating Scale
        0 (terrible fit) to 5 (perfect fit). Half-point increments are allowed.

        ### Response Format (JSON)
        {{
            "movie 1 (1999)": 4.5,
            "movie 2 (1992)": 3,
            ...
            "movie {max_recommendations} (1970)": 2
        }}

        Use each listed title exactly once, keep the original order, match the spelling (including release details), and return only the JSON object—no commentary or code fences.
        """
    ).strip()

    response = await query_llm(prompt, evaluation_session)
    ratings = _parse_rating_response(response, top_titles)
    missing = tuple(title for title in top_titles if title not in ratings)
    mean_rating = round(sum(ratings.values()) / len(top_titles), 3) if ratings else 0

    return {
        "recommendations": top_titles,
        "ratings": ratings,
        "mean_rating": mean_rating,
        "missing_titles": missing,
        "raw_response": response,
        "questioner_session": run_history.questioner_session,
        "answerer_session": run_history.answerer_session,
        "eval_session": evaluation_session,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from history import deserialise_run_record
    from tqdm import tqdm
    import polars as pl
    from tasks.movie_lens.data import load_balanced_dataset

    movies = load_balanced_dataset(fraction=0.25)

    parser = argparse.ArgumentParser(prog="Experiment evaluator")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="List of directories to evaluate",
    )
    parser.add_argument(
        "--questioner-input-price",
        type=float,
        default=0.28,
        help="Questioner input token price per 1M tokens",
    )
    parser.add_argument(
        "--questioner-output-price",
        type=float,
        default=0.2,
        help="Questioner output token price per 1M tokens",
    )
    parser.add_argument(
        "--answerer-input-price",
        type=float,
        default=0.28,
        help="Answerer input token price per 1M tokens",
    )
    parser.add_argument(
        "--answerer-output-price",
        type=float,
        default=0.2,
        help="Answerer output token price per 1M tokens",
    )
    args = parser.parse_args()

    paths: list[Path] = args.paths
    questioner_input_price: float = args.questioner_input_price
    questioner_output_price: float = args.questioner_output_price
    answerer_input_price: float = args.answerer_input_price
    answerer_output_price: float = args.answerer_output_price

    import asyncio

    results = []

    async def main():
        results = []
        for dir_path in paths:
            run_evals: list[MovieLensEval] = []
            for i, path in enumerate(
                tqdm(dir_path.rglob("*run.json"), desc=f"Loading {dir_path.name}")
            ):
                eval_session = LLMRequestSession("gpt_5")
                with path.open("r", encoding="utf-8") as f:
                    run_record = deserialise_run_record(json.load(f))
                run_evals.append(
                    await get_movielens_eval(
                        run_record, movies[i].persona, eval_session
                    )
                )

            group_eval = get_movielens_group_eval(run_evals)
            cost = {
                "questioner_input_price": (questioner_input_price / 1_000_000)
                * group_eval["questioner_input_tokens"],
                "questioner_output_price": (questioner_output_price / 1_000_000)
                * group_eval["questioner_output_tokens"],
                "answerer_input_price": (answerer_input_price / 1_000_000)
                * group_eval["answerer_input_tokens"],
                "answerer_output_price": (answerer_output_price / 1_000_000)
                * group_eval["answerer_output_tokens"],
                "eval_input_price": (1.25 / 1_000_000)
                * group_eval["eval_input_tokens"],
                "eval_output_price": (10 / 1_000_000)
                * group_eval["eval_output_tokens"],
            }

            results.append({**group_eval, **cost})

        df = pl.DataFrame(results)
        df.write_csv("movies_eval.csv")

        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON TABLE")
        print("=" * 100)

        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(df)

    asyncio.run(main())
