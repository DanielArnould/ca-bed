import re
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Sequence, TypedDict

from history import RunRecord
from models import LLMRequestSession, query_llm
from tasks.movie_lens.common import PersonaContext, format_persona_context


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


class MovieLensGroupEval(TypedDict):
    num_runs: int
    expected_ratings: int
    captured_ratings: int
    rating_coverage: float
    overall_mean_rating: float | None
    mean_mean_rating: float | None
    missing_titles: dict[str, int]


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


_RATING_LINE_PATTERN = re.compile(r"\d+\.\s*([^|]+)\|([^\n]+)")


def _select_top_titles(
    belief_state: dict[str, float], limit: int
) -> tuple[str, ...]:
    ordered = sorted(
        belief_state.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return tuple(title for title, _score in ordered[:limit])


def _parse_rating_response(
    response: str, expected_titles: Sequence[str]
) -> dict[str, float]:
    matches = _RATING_LINE_PATTERN.findall(response)
    if not matches:
        raise ValueError(
            "Could not locate numbered Title|Score ratings in evaluation response"
        )

    lookup = {title.casefold(): title for title in expected_titles}
    ratings: dict[str, float] = {}

    for raw_title, raw_score in matches:
        title = raw_title.strip().strip('"\'' "\u201c\u201d")
        score_text = raw_score.strip()

        if not title or not score_text:
            continue

        mapped = lookup.get(title.casefold())
        if mapped is None or mapped in ratings:
            continue

        score_match = re.search(r"[-+]?\d+(?:\.\d+)?", score_text)
        if not score_match:
            continue

        numeric = float(score_match.group(0))
        ratings[mapped] = numeric

    return ratings


def get_movielens_group_eval(
    movie_evals: Sequence[MovieLensEval],
) -> MovieLensGroupEval:
    total_expected = sum(len(movie_eval["recommendations"]) for movie_eval in movie_evals)
    total_captured = sum(len(movie_eval["ratings"]) for movie_eval in movie_evals)
    coverage = total_captured / total_expected if total_expected else 0.0

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
        "rating_coverage": round(coverage, 3),
        "overall_mean_rating": overall_mean,
        "mean_mean_rating": mean_of_means,
        "missing_titles": dict(sorted(missing_counter.items())),
    }


async def get_movielens_eval(
    run_history: RunRecord,
    persona: PersonaContext,
    evaluation_session: LLMRequestSession,
    max_recommendations: int = 10,
) -> MovieLensEval:
    top_titles = _select_top_titles(run_history.final_belief_state, max_recommendations)

    if not top_titles:
        return {
            "recommendations": tuple(),
            "ratings": {},
            "mean_rating": None,
            "missing_titles": tuple(),
            "raw_response": "",
        }

    persona_block = format_persona_context(persona)
    movies_block = "\n".join(
        f"{idx}. {title}" for idx, title in enumerate(top_titles, start=1)
    )

    prompt = dedent(
        f"""\
        You are scoring how well a recommendation list suits a specific movie-goer persona.

        ### Persona Briefing
        {persona_block}

        ### Candidate Films
        {movies_block}

        Rate each candidate film for this persona from 1 (terrible fit) to 5 (perfect fit).
        You may use half-point increments. Base your score on the persona's tastes, motifs,
        aversions, and reference favourites/dislikes. If you are uncertain about a film,
        make the best informed estimate.

        ### Response Format
        One line per film:
        1. <Exact Title>|<score>
        2. <Exact Title>|<score>
        ...
        {len(top_titles)}. <Exact Title>|<score>

        Use each listed title exactly once, matching its spelling (including release details), and provide only the numbered lines—no commentary or extra text.
        """
    ).strip()

    response = await query_llm(prompt, evaluation_session)

    ratings = _parse_rating_response(response, top_titles)

    missing = tuple(title for title in top_titles if title not in ratings)
    mean_rating = (
        round(sum(ratings.values()) / max_recommendations, 3)
        if ratings
        else None
    )

    return MovieLensEval(**{
        "recommendations": top_titles,
        "ratings": ratings,
        "mean_rating": mean_rating,
        "missing_titles": missing,
        "raw_response": response,
    })

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from history import deserialise_run_record
    import json
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
        default=0.05,
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
        default=0.05,
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
            for i, path in enumerate(tqdm(dir_path.rglob("*run.json"), desc=f"Loading {dir_path.name}")):
                eval_session = LLMRequestSession('gpt_5')
                with path.open("r", encoding="utf-8") as f:
                    run_record = deserialise_run_record(json.load(f))
                run_evals.append(await get_movielens_eval(run_record, movies[i].persona, eval_session))
            results.append(get_movielens_group_eval(run_evals))

        df = pl.DataFrame(results)

        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON TABLE")
        print("=" * 100)

        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(df)

    asyncio.run(main())
