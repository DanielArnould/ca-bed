from dataclasses import dataclass
from enum import Enum, auto
from dotenv import load_dotenv
import os

from openai import OpenAI

load_dotenv()


class Model(Enum):
    DEEPSEEK_CHAT = auto()
    DEEPSEEK_REASONER = auto()


DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")


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


def _call_deepseek_chat(input_text: str) -> LLMOutput:
    assert DEEPSEEK_KEY is not None, "DEEPSEEK_KEY not found in environment!"
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": input_text}],
        logprobs=True,
        max_tokens=4096,
        temperature=1.0,
        stream=False,
    )

    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
        tokens=[
            Token(token.token, token.bytes, token.logprob)  # type: ignore
            for token in response.choices[0].logprobs.content  # type: ignore
        ],
    )


def _call_deepseek_reasoner(input_text: str) -> LLMOutput:
    assert DEEPSEEK_KEY is not None, "DEEPSEEK_KEY not found in environment!"
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=32_000,
    )

    return LLMOutput(
        string=response.choices[0].message.content,  # type: ignore
        reasoning=response.choices[0].message.reasoning_content,  # type: ignore
    )


def call_llm(input_text: str, model: Model) -> LLMOutput:
    match model:
        case Model.DEEPSEEK_CHAT:
            return _call_deepseek_chat(input_text)
        case Model.DEEPSEEK_REASONER:
            return _call_deepseek_reasoner(input_text)
