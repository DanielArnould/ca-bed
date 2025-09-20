from enum import Enum
import logging
from dotenv import load_dotenv
import os

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()
LOGGER = logging.getLogger("LLM Models")
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0


def _make_async_client(api_key: str | None, base_url: str | None) -> AsyncOpenAI | None:
    if api_key is None:
        print(
            f"Skipping client creation for {base_url if base_url is not None else 'OpenAI'}"
        )
        return None

    print(
        f"Creating client for {base_url if base_url is not None else 'OpenAI'} with key {len(api_key[:-1]) * '#' + api_key[-4:]}"
    )
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


CLIENTS = {
    "deepseek": _make_async_client(
        os.getenv("DEEPSEEK_KEY"), "https://api.deepseek.com"
    ),
    "together": _make_async_client(
        os.getenv("TOGETHER_AI_KEY"), "https://api.together.xyz/v1"
    ),
    "openai": _make_async_client(os.getenv("OPENAI_KEY"), None),
    "ollama": _make_async_client("ollama", "http://localhost:11434/v1"),
}


class Model(Enum):
    DEEPSEEK_CHAT = ("deepseek", "deepseek-chat", {})
    DEEPSEEK_CHAT_TOGETHER_AI = ("together", "deepseek-ai/DeepSeek-V3.1", {})
    GPT_4O_MINI = ("openai", "gpt-4o-mini-2024-07-18", {})
    LLAMA_3_3_70B = ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", {})
    LLAMA_3_2_3B = ("together", "meta-llama/Llama-3.2-3B-Instruct-Turbo", {})
    GEMMA_3N_4B_TOGETHER_AI = ("together", "google/gemma-3n-E4B-it", {})
    GEMMA_3N_4B_OLLAMA = ("ollama", "gemma3n:e4b", {})
    GPT_OSS_20B = ("together", "openai/gpt-oss-20b", {"reasoning_effort": "low"})
    DUMMY = ("dummy", "dummy", {})


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=3, max=360))
async def call_llm(input_text: str, model: Model) -> tuple[str, int, int]:
    """Returns (response, prompt token count, completion token count)"""

    if model == Model.DUMMY:
        print(f"[DUMMY LLM]: {input_text}")
        return input("Enter a response: "), 0, 0

    client_key, model_id, kwargs = model.value
    client = CLIENTS[client_key]
    if client is None:
        raise RuntimeError(
            f"Client for {model.name} is not configured with an API key."
        )

    LOGGER.info(f"Sending to {model_id}: '{input_text.replace('\n', ' ')}'")

    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=4096,
        temperature=0,
        n=1,
        stream=False,
        **kwargs,  # type: ignore
    )

    LOGGER.info(f"Received response from {model_id}: '{response}'")
    return (
        response.choices[0].message.content,  # type: ignore
        response.usage.prompt_tokens,  # type: ignore
        response.usage.completion_tokens,  # type: ignore
    )
