from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

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


def load_all_data() -> list[MedDGInstance]:
    data_path = Path(os.path.dirname(__file__), "MedDG.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        MedDGInstance(disease=inst["target"], self_report=inst["self_repo"])
        for inst in data
    ]


def load_balanced_data(
    sample_percentage: float = 1.0, seed: int = 42
) -> list[MedDGInstance]:
    data_path = Path(os.path.dirname(__file__), "MedDG.json")
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    disease_groups = defaultdict(list)
    for item in raw:
        disease = item["target"]
        self_report = item["self_repo"]
        disease_groups[disease].append({"disease": disease, "self_report": self_report})

    total_samples = sum(len(group) for group in disease_groups.values())
    target_size = int(total_samples * sample_percentage)
    base_samples_per_disease = target_size // len(disease_groups)

    # First pass: allocate what each disease can provide
    # keep track of classes that are limited in size
    allocation = {}
    leftover = 0

    for disease, items in disease_groups.items():
        available = len(items)
        desired = base_samples_per_disease

        if available < desired:
            # This disease is limited by its size
            allocation[disease] = available
            leftover += desired - available
        else:
            # This disease can provide the desired amount
            allocation[disease] = desired

    # Second pass: redistribute leftover to diseases that can take more
    # so we are still close to the target subset size, but also balanced
    if leftover > 0:
        diseases_with_capacity = [
            disease
            for disease, items in disease_groups.items()
            if len(items) > allocation[disease]
        ]

        while leftover > 0 and diseases_with_capacity:
            for disease in diseases_with_capacity[:]:
                if leftover == 0:
                    break

                available = len(disease_groups[disease])
                current_allocation = allocation[disease]

                if current_allocation < available:
                    allocation[disease] += 1
                    leftover -= 1
                else:
                    # This disease is now at capacity
                    diseases_with_capacity.remove(disease)

    random.seed(seed)
    balanced_data = []

    for disease, items in disease_groups.items():
        n_samples = allocation[disease]
        if n_samples > 0:
            sampled_items = random.sample(items, n_samples)
            balanced_data.extend(sampled_items)

    random.shuffle(balanced_data)

    return [
        MedDGInstance(disease=item["disease"], self_report=item["self_report"])
        for item in balanced_data
    ]


if __name__ == "__main__":
    from collections import Counter

    dataset = load_balanced_data(0.5)
    print(Counter(item.disease for item in dataset))
