import json
import os
from pathlib import Path
from typing import TypedDict, cast


class VictimInformation(TypedDict):
    name: str
    introduction: str
    cause_of_death: str
    murder_weapon: str


class SuspectInformation(TypedDict):
    name: str
    introduction: str
    relationship: str
    reason_at_scene: str
    suspicion: str
    motive: str
    opportunity: str
    story: str
    testimony: str
    is_murderer: bool


class DetectiveCasesInstance(TypedDict):
    num: int
    time: str
    location: str
    victim: VictimInformation
    suspects: list[SuspectInformation]


def load_all_data() -> list[DetectiveCasesInstance]:
    data_path = Path(os.path.dirname(__file__), "DetectiveCases.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return cast(list[DetectiveCasesInstance], data)
