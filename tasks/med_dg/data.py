from dataclasses import dataclass
import json
import os
from pathlib import Path
import polars as pl
import polars as pl

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


def load_data_even(
    sample_percentage: float = 1.0, seed: int = 42
) -> list[MedDGInstance]:
    if not (0 < sample_percentage <= 1):
        raise ValueError("sample_percentage must be in (0, 1].")

    data_path = Path(os.path.dirname(__file__), "MedDG.json")
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pl.DataFrame(raw).rename({"self_repo": "self_report", "target": "disease"})
    if sample_percentage == 1.0:
        return [
            MedDGInstance(disease=row["disease"], self_report=row["self_report"])
            for row in df.iter_rows(named=True)
        ]

    total = df.height
    diseases = sorted(df.select(pl.col("disease")).unique().to_series().to_list())
    k = len(diseases)
    budget = int(round(total * sample_percentage))
    if budget == 0:
        return []

    size_map: dict[str, int] = (
        df.group_by("disease").agg(pl.len().alias("cnt")).to_dict(as_series=False)
    )
    size_map = {d: c for d, c in zip(size_map["disease"], size_map["cnt"])}
    base = budget // k
    remainder = budget % k
    alloc: dict[str, int] = {}
    for idx, d in enumerate(diseases):
        alloc[d] = base + (1 if idx < remainder else 0)

    # Cap by class capacity, track leftover
    leftover = 0
    for d in diseases:
        cap = size_map[d]
        if alloc[d] > cap:
            leftover += alloc[d] - cap
            alloc[d] = cap

    # Redistribute leftover to classes with remaining capacity
    if leftover > 0:
        for d in diseases:
            if leftover == 0:
                break
            cap = size_map[d]
            while alloc[d] < cap and leftover > 0:
                alloc[d] += 1
                leftover -= 1

    # Now sample rows per disease
    def sample_group(g: pl.DataFrame) -> pl.DataFrame:
        d = g["disease"][0]
        n = alloc.get(d, 0)
        if n <= 0:
            return pl.DataFrame(schema=g.schema)
        if g.height <= n:
            return g
        return g.sample(n=n, seed=seed)

    sampled = df.group_by("disease").map_groups(sample_group)
    # Drop any empty groups
    sampled = sampled.filter(pl.col("self_report").is_not_null())
    # Deterministic shuffle
    sampled = sampled.sample(fraction=1.0, seed=seed)

    return [
        MedDGInstance(disease=row["disease"], self_report=row["self_report"])  # type: ignore[index]
        for row in sampled.iter_rows(named=True)
    ]


if __name__ == "__main__":
    from collections import Counter

    dataset = load_data_even(0.3)
    counter = Counter()
    for item in dataset:
        counter[item.disease] += 1

    print(counter)
