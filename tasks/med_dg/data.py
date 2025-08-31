from dataclasses import dataclass
import json
import os
from pathlib import Path

MED_DG_SET = [
    "Enteritis",
    "Gastritis",
    "Gastroenteritis",
    "Esophagitis",
    "Cholecystitis",
    "Appendicitis",
    "Pancreatitis",
    "Gastric ulcer",
    "Constipation",
    "Cold",
    "Irritable bowel syndrome",
    "Diarrhea",
    "Allergic rhinitis",
    "Upper respiratory tract infection",
    "Pneumonia",
]


@dataclass
class MedDGInstance:
    disease: str
    self_report: str


def load_data() -> list[MedDGInstance]:
    data_path = Path(os.path.dirname(__file__), "MedDG.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        MedDGInstance(disease=inst["target"], self_report=inst["self_repo"])
        for inst in data
    ]
