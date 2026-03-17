from asyncio import Semaphore
from dataclasses import dataclass
import os

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI


load_dotenv()
CLIENT = AsyncOpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE_URL"))
SEMAPHORE = Semaphore(int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))


@dataclass(slots=True)
class LLM:
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0


async def get_response(
    prompt: str,
    llm: LLM,
    task_id: str,
) -> str:
    async with SEMAPHORE:
        logger.bind(task_id=task_id).info(f"Sending prompt: '{prompt}'")
        response = await CLIENT.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=1.0,
        )

    if response.usage is None:
        raise ValueError(
            f"No response usage metrics for {llm} for the prompt: '{prompt}'"
        )

    prompt_tokens = response.usage.prompt_tokens
    llm.total_input_tokens += prompt_tokens

    completion_tokens = response.usage.completion_tokens
    llm.total_output_tokens += completion_tokens

    content = response.choices[0].message.content
    if content is None:
        raise ValueError(f"No content for {llm} for the prompt: '{prompt}'")

    logger.bind(task_id=task_id).info(f"Received response: '{content}'")
    return content
