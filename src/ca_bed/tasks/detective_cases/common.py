from ca_bed.tasks.detective_cases.data import DetectiveCasesInstance


def build_case_context(case: DetectiveCasesInstance) -> str:
    parts = [
        f"Time: {case['time']}",
        f"Location: {case['location']}",
        f"Victim: {case['victim']['name']} - {case['victim']['introduction']} (Cause of death: {case['victim']['cause_of_death']}, Weapon: {case['victim']['murder_weapon']})",
        "\nSuspects:",
    ]
    for s in case["suspects"]:
        parts.append(
            f"- {s['name']}: {s['introduction']}. "
            f"Reason at scene: {s['reason_at_scene']} "
            f"Testimony: {s['testimony']}"
        )
    return "\n".join(parts)
