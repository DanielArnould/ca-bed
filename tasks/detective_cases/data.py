import json
import os
from pathlib import Path
from typing import TypedDict, cast


class VictimInformation(TypedDict):
    name: str
    introduction: str
    cause_of_death: str
    murder_weapon: str


class SuspectOverview(TypedDict):
    name: str
    introduction: str


class TimelineEvent(TypedDict):
    time: str
    activity: str


class SuspectInformation(TypedDict):
    name: str
    introduction: str
    relationship: str
    reason_at_scene: list[str]
    suspicion: list[str]
    motive: list[str]
    opportunity: list[str]
    access_to_weapon: list[str]
    is_murderer: bool
    evidence: str
    testimony: list[str]
    timeline: list[TimelineEvent]
    story: str
    task: str


class InitialInformation(TypedDict):
    time: str
    location: str
    victim: VictimInformation
    suspect: list[SuspectOverview]


class DetectiveCasesInstance(TypedDict):
    time: str
    location: str
    victim: VictimInformation
    suspects: list[SuspectInformation]
    initial_information: InitialInformation
    murderer: str
    explanation: str


def load_all_data() -> list[DetectiveCasesInstance]:
    data_path = Path(os.path.dirname(__file__), "ARBenchTest.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return cast(list[DetectiveCasesInstance], data)


if __name__ == "__main__":
    data = load_all_data()
    print(len(data))
