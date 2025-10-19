import math
import string
from textwrap import dedent
from typing import Any, override

from tenacity import retry, stop_after_attempt

from models import llm_models
from .bayesian import Bayesian
import logging

logger = logging.getLogger("Bayesian LogProbs")

class BayesianLogProbs(Bayesian):
    def __str__(self) -> str:
        return (
            "Twenty Questions (Bayesian LogProbs): "
            f"{self.questioner_session.model_key=} "
            f"{self.answerer_session.model_key=} "
            f"{self.task_answer=} "
            f"{self.max_question_nodes=} "
            f"{self.max_lookahead_depth=} "
            f"{self.max_conversation_depth=} "
            f"{self.confidence_threshold=} "
            f"{self.hypothesis_space=}"
        )

    @override
    @retry(stop=stop_after_attempt(2))
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        model_config = llm_models.get(self.questioner_session.model_key)
        if model_config is None:
            raise KeyError(
                f"Unknown model key: {self.questioner_session.model_key}. "
                f"Available models are: {list(llm_models.keys())}"
            )

        client = model_config.get("client")
        if client is None:
            raise RuntimeError(
                "The configured questioner model does not expose an API client, "
                "so token logprobs cannot be retrieved."
            )

        params = dict(model_config.get("params", {}))
        params.update(
            {
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 20,
                'temperature': 1
            }
        )

        positive_label = answers[0] if answers else "Yes"
        negative_label = (
            answers[1]
            if len(answers) > 1
            else ("No" if answers else "No")
        )

        likelihoods: dict[str, dict[str, float]] = {}

        for hypothesis in hypotheses:
            prompt = self._build_hypothesis_prompt(
                hypothesis=hypothesis,
                question=question,
                positive_label=positive_label,
                negative_label=negative_label,
            )

            response = await client.chat.completions.create(
                model=model_config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                **params,
            )

            self._update_usage(response)

            yes_prob, no_prob = self._extract_binary_probabilities(
                response=response,
                positive_label=positive_label,
                negative_label=negative_label,
            )

            likelihoods[hypothesis] = {
                positive_label: yes_prob,
                negative_label: no_prob,
            }

        missing = set(hypotheses) - set(likelihoods)
        if missing:
            raise RuntimeError(f"Missing likelihoods for: {missing}")

        return likelihoods

    @staticmethod
    def _build_hypothesis_prompt(
        hypothesis: str,
        question: str,
        positive_label: str,
        negative_label: str,
    ) -> str:
        return dedent(
            f"""\
            You are playing a game of 20 Questions and you know with certainty that the secret entity X is "{hypothesis}".

            Answer the following question *truthfully* as that entity.

            ### Instructions
            - Base your answer solely on the truthful properties of "{hypothesis}".
            - Respond using only a single word: "{positive_label}" or "{negative_label}" EXACTLY, DO NOT DEVIATE.
            - Do not include punctuation or additional commentary.

            ### Question
            "{question}"
            """
        ).strip()

    def _extract_binary_probabilities(
        self,
        response: Any,
        positive_label: str,
        negative_label: str,
    ) -> tuple[float, float]:
        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None or not getattr(logprobs, "content", None):
            raise RuntimeError("Model response did not include token logprobs.")

        tokens = logprobs.content
        if not tokens:
            raise RuntimeError("Token logprobs payload was empty.")

        token_block = tokens[0]
        logprob_map: dict[str, float] = {}

        def register(token: str, logprob: float) -> None:
            normalised = self._normalise_token(token)
            if not normalised:
                return
            current = logprob_map.get(normalised)
            if current is None or logprob > current:
                logprob_map[normalised] = logprob

        register(token_block.token, token_block.logprob)
        for alternative in getattr(token_block, "top_logprobs", []) or []:
            register(alternative.token, alternative.logprob)

        yes_logprob = logprob_map.get(self._normalise_token(positive_label))
        no_logprob = logprob_map.get(self._normalise_token(negative_label))

        if yes_logprob is not None and no_logprob is not None:
            yes_prob, no_prob = self._normalise_binary_logprobs(
                yes_logprob, no_logprob
            )
        elif yes_logprob is not None:
            yes_prob = self._prob_from_logprob(yes_logprob)
            yes_prob, no_prob = self._apply_binary_epsilon(
                yes_prob, 1.0 - yes_prob
            )
        elif no_logprob is not None:
            no_prob = self._prob_from_logprob(no_logprob)
            yes_prob, no_prob = self._apply_binary_epsilon(
                1.0 - no_prob, no_prob
            )
        else:
            no_prob = 0.5
            yes_prob = 0.5

        logger.info(f'Probability (yes, no): {yes_prob, no_prob}')

        return yes_prob, no_prob

    @staticmethod
    def _normalise_binary_logprobs(
        yes_logprob: float, no_logprob: float
    ) -> tuple[float, float]:
        max_logprob = max(yes_logprob, no_logprob)
        yes_weight = math.exp(yes_logprob - max_logprob)
        no_weight = math.exp(no_logprob - max_logprob)
        total = yes_weight + no_weight
        if total <= 0:
            raise RuntimeError("Normalisation failed due to zero total weight.")

        yes_prob = yes_weight / total
        no_prob = no_weight / total

        epsilon = 1e-6
        yes_prob = min(max(yes_prob, epsilon), 1 - epsilon)
        no_prob = 1 - yes_prob
        return yes_prob, no_prob

    @staticmethod
    def _normalise_token(token: str) -> str:
        cleaned = (
            token.replace("Ġ", " ")
            .replace("▁", " ")
            .replace("\n", " ")
            .replace("\r", " ")
        )
        cleaned = cleaned.strip()
        cleaned = cleaned.strip(string.punctuation)
        return cleaned.lower()

    @staticmethod
    def _prob_from_logprob(logprob: float) -> float:
        return math.exp(max(min(logprob, 0.0), -100.0))

    @staticmethod
    def _apply_binary_epsilon(yes_prob: float, no_prob: float) -> tuple[float, float]:
        epsilon = 1e-6
        yes_prob = min(max(yes_prob, epsilon), 1 - epsilon)
        no_prob = min(max(no_prob, epsilon), 1 - epsilon)
        total = yes_prob + no_prob
        yes_prob /= total
        no_prob = 1 - yes_prob
        return yes_prob, no_prob

    def _update_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)

        self.questioner_session.total_input_tokens += prompt_tokens
        self.questioner_session.total_output_tokens += completion_tokens
