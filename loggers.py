from logging import Logger
import logging
import multiprocessing
from pathlib import Path


def setup_logger(name: str, output_dir: Path) -> Logger:
    process_name = multiprocessing.current_process().name

    logger = logging.getLogger(f"{name} - {process_name}")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(output_dir / f"{process_name}.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s]: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger(name: str) -> Logger:
    process_name = multiprocessing.current_process().name
    return logging.getLogger(f"{name} - {process_name}")
