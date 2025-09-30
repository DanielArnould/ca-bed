import logging
import os
from dataclasses import dataclass
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()

logger = logging.getLogger("LLM Models")


def _create_async_openai_client(
    api_key: str | None, api_base_url: str | None
) -> AsyncOpenAI | None:
    service_name = api_base_url or "OpenAI"

    if api_key is None:
        print(f"No API key found. Skipping client creation for {service_name}...")
        return None

    masked_key = f"{'#' * (len(api_key) - 4)}{api_key[-4:]}"
    print(f"Creating client for {service_name} with key {masked_key}")

    return AsyncOpenAI(api_key=api_key, base_url=api_base_url)


print("Creating API clients...")
_deepseek_client = _create_async_openai_client(
    os.getenv("DEEPSEEK_KEY"), "https://api.deepseek.com"
)
_together_client = _create_async_openai_client(
    os.getenv("TOGETHER_AI_KEY"), "https://api.together.xyz/v1"
)
_openai_client = _create_async_openai_client(os.getenv("OPENAI_KEY"), None)
_ollama_client = _create_async_openai_client("ollama", "http://localhost:11434/v1")
print("API client creation complete.")

llm_models: dict[str, dict] = {
    "deepseek_chat": {
        "client": _deepseek_client,
        "model_name": "deepseek-chat",
        "params": {},
    },
    "deepseek_together_ai": {
        "client": _together_client,
        "model_name": "deepseek-ai/DeepSeek-V3.1",
        "params": {},
    },
    "gpt_4o_mini": {
        "client": _openai_client,
        "model_name": "gpt-4o-mini-2024-07-18",
        "params": {"temperature": 0.0},
    },
    "llama_3_3_70b": {
        "client": _together_client,
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "params": {},
    },
    "llama_3_2_3b": {
        "client": _together_client,
        "model_name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "params": {},
    },
    "gemma_3n_4b_together_ai": {
        "client": _together_client,
        "model_name": "google/gemma-3n-E4B-it",
        "params": {},
    },
    "gemma_3n_4b_ollama": {
        "client": _ollama_client,
        "model_name": "gemma3n:e4b",
        "params": {},
    },
    "gpt_oss_20b": {
        "client": _together_client,
        "model_name": "openai/gpt-oss-20b",
        "params": {"reasoning_effort": "low", "max_tokens": 4096, "temperature": 0.0},
    },
    "dummy": {
        "client": None,
        "model_name": "dummy",
        "params": {},
    },
}


@dataclass
class LLMRequestSession:
    model_key: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self):
        if self.model_key not in llm_models:
            raise KeyError(
                f"Unknown model key: {self.model_key}. Available models are: {list(llm_models.keys())}"
            )


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=3, max=60))
async def query_llm(input_text: str, session: LLMRequestSession) -> str:
    if session.model_key == "dummy":
        print(f"[DUMMY LLM] Prompt: {input_text}")
        return input("Enter a mock response: ")

    start_time = time.perf_counter()
    model_config = llm_models[session.model_key]

    client: AsyncOpenAI | None = model_config["client"]
    model_name: str = model_config["model_name"]
    extra_params: dict = model_config["params"]

    if client is None:
        raise RuntimeError(f"No API client configured for model: {session.model_key}")

    logger.info(f"Sending to {model_name}: '{input_text.replace('\n', ' ')}'")

    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        stream=False,
        **extra_params,  # type: ignore
    )

    elapsed = time.perf_counter() - start_time
    prompt_tokens = response.usage.prompt_tokens  # type: ignore
    completion_tokens = response.usage.completion_tokens  # type: ignore
    session.total_input_tokens += prompt_tokens
    session.total_output_tokens += completion_tokens

    logger.info(
        f"Received response from {model_name} in {elapsed}s with {prompt_tokens} input tokens and {completion_tokens} output tokens: '{response}'"
    )

    return response.choices[0].message.content  # type: ignore
