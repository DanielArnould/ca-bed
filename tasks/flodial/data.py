from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass
class FlodialInstance:
    self_report: str
    target: str


def load_data() -> list[FlodialInstance]:
    data_path = Path(os.path.dirname(__file__), "flodial.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        FlodialInstance(self_report=instance["self_repo"], target=instance["target"])
        for instance in data
    ]


def get_hypothesis_space():
    data = load_data()
    get_hypothesis_space = list(set([ins.target for ins in data]))
    return get_hypothesis_space
        
get_hypothesis_space()