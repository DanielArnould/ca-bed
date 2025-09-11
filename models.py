from dataclasses import dataclass
from enum import Enum, auto
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


class Model(Enum):
    DEEPSEEK_CHAT = auto()
    DEEPSEEK_CHAT_TOGETHER_AI = auto()
    DEEPSEEK_REASONER = auto()
    DUMMY = auto()
    GPT_4O_MINI = auto()
    GPT_5_NANO = auto()


DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
DEEPSEEK_CLIENT = (
    AsyncOpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_KEY is not None
    else None
)

TOGETHER_AI_KEY = os.getenv("TOGETHER_AI_KEY")
TOGETHER_AI_CLIENT = (
    AsyncOpenAI(api_key=TOGETHER_AI_KEY, base_url="https://api.together.xyz/v1")
    if TOGETHER_AI_KEY is not None
    else None
)

OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_KEY) if OPENAI_KEY is not None else None


@dataclass
class Token:
    string: str
    bytes: list[int]
    logprob: float


@dataclass
class LLMOutput:
    string: str
    tokens: list[Token] | None = None
    reasoning: str | None = None


async def _call_deepseek_chat(input_text: str) -> LLMOutput:
    assert DEEPSEEK_CLIENT is not None, (
        "DEEPSEEK_CLIENT not setup (have you provided a key?)"
    )
    LOGGER.info(f"Sending message to Deepseek Chat, {input_text.replace('\n', '')}")

    response = await DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": input_text}],
        logprobs=True,
        max_tokens=4096,
        temperature=0.0,  # UoT does not specify a temperature, but their repo suggests 0.0
        stream=False,
    )

    LOGGER.info(f"Received response from Deepseek Chat {response}")
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
        tokens=[
            Token(token.token, token.bytes, token.logprob)  # type: ignore
            for token in response.choices[0].logprobs.content  # type: ignore
        ],
    )


async def _call_deepseek_reasoner(input_text: str) -> LLMOutput:
    assert DEEPSEEK_CLIENT is not None, (
        "DEEPSEEK_CLIENT not setup (have you provided a key?)"
    )
    LOGGER.info(f"Sending message to Deepseek Reasoner, {input_text.replace('\n', '')}")

    response = await DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=32_000,
    )

    LOGGER.info(f"Received response from Deepseek Chat {response}")
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
        reasoning=response.choices[0].message.reasoning_content,  # type: ignore
    )


async def _call_dummy(input_text: str) -> LLMOutput:
    # NOT ASYNCHRONOUS
    print(f"[DUMMY LLM]: {input_text}")
    response = input("Enter a response: ")
    return LLMOutput(response)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=3, max=360))
async def _call_gpt_4o_mini(input_text: str) -> LLMOutput:
    assert OPENAI_CLIENT is not None, (
        "OPENAI_CLIENT not setup (have you provided a key?)"
    )
    LOGGER.info(f"Sending message to gpt-4o-mini, {input_text.replace('\n', '')}")

    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=4096,
        temperature=0.0,
        n=1,
    )

    LOGGER.info(f"Received response from gpt-4o-mini {response}")
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
    )


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=3, max=360))
async def _call_gpt_5_nano(input_text: str) -> LLMOutput:
    assert OPENAI_CLIENT is not None, (
        "OPENAI_CLIENT not setup (have you provided a key?)"
    )
    LOGGER.info(f"Sending message to gpt-5-nano, {input_text.replace('\n', '')}")

    response = await OPENAI_CLIENT.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[{"role": "user", "content": input_text}],
    )

    LOGGER.info(f"Received response from gpt-5-nano: {response}")
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
    )


async def _call_deepseek_chat_together_ai(input_text: str) -> LLMOutput:
    assert TOGETHER_AI_CLIENT is not None, (
        "TOGETHER_AI_CLIENT not setup (have you provided a key?)"
    )
    LOGGER.info(
        f"Sending message to Deepseek V3.1 (Together AI): {input_text.replace('\n', '')}"
    )

    response = await TOGETHER_AI_CLIENT.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.1",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=4096,
        temperature=0.0,
        stream=False,
    )

    LOGGER.info(f"Received response from Deepseek V3.1 (Together AI) {response}")
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
    )


async def call_llm(input_text: str, model: Model) -> LLMOutput:
    match model:
        case Model.DEEPSEEK_CHAT:
            return await _call_deepseek_chat(input_text)
        case Model.DEEPSEEK_REASONER:
            return await _call_deepseek_reasoner(input_text)
        case Model.DUMMY:
            return await _call_dummy(input_text)
        case Model.GPT_4O_MINI:
            return await _call_gpt_4o_mini(input_text)
        case Model.GPT_5_NANO:
            return await _call_gpt_5_nano(input_text)
        case Model.DEEPSEEK_CHAT_TOGETHER_AI:
            return await _call_deepseek_chat_together_ai(input_text)
