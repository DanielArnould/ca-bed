from dataclasses import dataclass
from typing import Tuple
import json

@dataclass
class CraftMDInstance:
    patient_info: str
    ground_truth: str
    atomic_facts: list[str]

def load_data() -> Tuple[list[CraftMDInstance], list[str]]:
    with open(f'tasks/craft_md/all_craft_md.jsonl', "r") as f:
        lines = f.readlines()

    output_data = [json.loads(line) for line in lines]

    unique_answers = set()
    parsed = []
    for data in output_data:
        patient_info = data['context'][0]
        ground_truth = data['answer'].lower().strip()
        atomic_facts = data['facts']

        unique_answers.add(ground_truth)
        parsed.append(CraftMDInstance(patient_info, ground_truth, atomic_facts))

    return parsed, list(unique_answers)

if __name__ == '__main__':
    with open(f'tasks/craft_md/all_craft_md.jsonl', "r") as f:
        lines = f.readlines()

    output_data = [json.loads(line) for line in lines]

    unique_answers = set()
    
    print(json.dumps(output_data[0], indent=4))

    for data in output_data:
        unique_answers.add(data['answer'].lower().strip())

    for option in unique_answers:
        print(option)

    print(len(unique_answers))