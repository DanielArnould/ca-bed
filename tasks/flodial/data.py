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


def test():
    data = load_data()
    print(len(data))
    for instance in data:
        print(f'issue: {instance.self_report} - cause - {instance.target}')
        
test()