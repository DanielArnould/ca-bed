from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass
class FlodialInstance:
    issue: str
    cause: str


def load_data() -> list[FlodialInstance]:
    data_path = Path(os.path.dirname(__file__), "flodial.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        FlodialInstance(issue=issue, cause=cause)
        for issue, cause in data.items()
    ]


def test():
    data = load_data()
    print(len(data))
    for instance in data:
        print(f'issue: {instance.issue} - cause - {instance.cause}')
        
# test()


def test2():
    trouble_name = {"a":"b"}
    print(trouble_name.values())
test2()