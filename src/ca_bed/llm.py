from asyncio import Semaphore
from dataclasses import dataclass
import os

from async_lru import alru_cache
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI


load_dotenv()
CLIENT = AsyncOpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE_URL"))
SEMAPHORE = Semaphore(int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))


@dataclass(slots=True, frozen=True)
class LLM:
    model: str


@alru_cache(maxsize=None)
async def get_response(
    prompt: str,
    llm: LLM,
) -> str:
    async with SEMAPHORE:
        logger.info(f"Sending prompt: '{prompt}'")
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

    content = response.choices[0].message.content
    if content is None:
        raise ValueError(f"No content for {llm} for the prompt: '{prompt}'")

    logger.info(f"Received response: '{content}'")
    return content
