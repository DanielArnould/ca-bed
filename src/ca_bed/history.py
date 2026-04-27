from dataclasses import dataclass

from ca_bed.node import EvidenceNode
from ca_bed.tasks.task import Task


@dataclass(slots=True, frozen=True)
class RunRecord:
    task: Task
    final_path: list[EvidenceNode]
