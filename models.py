from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv
import os

from openai import AsyncOpenAI

from loggers import get_logger

load_dotenv()


class Model(Enum):
    DEEPSEEK_CHAT = auto()
    DEEPSEEK_REASONER = auto()
    DUMMY = auto()


DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
DEEPSEEK_CLIENT = (
    AsyncOpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_KEY is not None
    else None
)


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
    logger = get_logger("LLM Models")
    logger.info("Sending message to Deepseek Chat")

    response = await DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": input_text}],
        logprobs=True,
        max_tokens=4096,
        temperature=0.0,  # UoT does not specify a temperature, but their repo suggests 0.0
        stream=False,
    )

    logger.info("Received response from Deepseek Chat")
    logger.debug(str(response))
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
    logger = get_logger("LLM Models")
    logger.info("Sending message to Deepseek Reasoner")

    response = await DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=32_000,
    )

    logger.info("Received response from Deepseek Chat")
    logger.debug(str(response))
    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
        reasoning=response.choices[0].message.reasoning_content,  # type: ignore
    )


async def _call_dummy(input_text: str) -> LLMOutput:
    # NOT ASYNCHRONOUS
    print(f"[DUMMY LLM]: {input_text}")
    response = input("Enter a response: ")
    return LLMOutput(response)


async def call_llm(input_text: str, model: Model) -> LLMOutput:
    match model:
        case Model.DEEPSEEK_CHAT:
            return await _call_deepseek_chat(input_text)
        case Model.DEEPSEEK_REASONER:
            return await _call_deepseek_reasoner(input_text)
        case Model.DUMMY:
            return await _call_dummy(input_text)
