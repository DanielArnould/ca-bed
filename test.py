from loguru import logger


logger.add
logger.info("Hi!")
logger.bind(task="Something").info("Bye!")
