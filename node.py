"""
This module contains the key constructs for the tree structure needed for enhanced UoT.
The tree structure ought to be as extendable as possible, but reward functions
may need to be differentiated and selected by the method.
"""

from dataclasses import dataclass, field

type Probability = float
type Item = str


@dataclass
class EvidenceNode:
    answer: str
    belief_state: list[tuple[Item, Probability]]
    parent: "QuestionNode | None" = None
    children: list["QuestionNode"] = field(default_factory=list)


@dataclass
class QuestionNode:
    question: str
    parent: EvidenceNode
    children: list[EvidenceNode] = field(default_factory=list)


def stringify(root: EvidenceNode) -> str:
    lines = []

    def _build_string(
        node: EvidenceNode | QuestionNode, prefix: str = "", is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "

        match node:
            case EvidenceNode(answer, item_probabilities, _, children_groups):
                formatted_item_probabilities = ", ".join(
                    [f"{item}: {prob}" for item, prob in item_probabilities]
                )
                lines.append(
                    f"{prefix}{connector}Answer: {answer} | Probabilities: [{formatted_item_probabilities}]"
                )

                for i, child in enumerate(children_groups):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children_groups) - 1)
            case QuestionNode(question, _, answer_nodes):
                lines.append(f"{prefix}{connector}Question: {question}")

                for i, child in enumerate(answer_nodes):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(answer_nodes) - 1)

    formatted_item_probabilities = ", ".join(
        [f"{item}: {prob}" for item, prob in root.belief_state]
    )
    lines.append(
        f"Answer: {root.answer} | Probabilities: [{formatted_item_probabilities}]"
    )
    for i, child in enumerate(root.children):
        _build_string(child, "", i == len(root.children) - 1)

    return "\n".join(lines)


if __name__ == "__main__":
    root = EvidenceNode(
        "ROOT", belief_state=[("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)]
    )

    question1 = QuestionNode("Is it greater than or equal to B?", parent=root)
    question1_affirmative = EvidenceNode(
        "Yes", belief_state=[("A", 0.5), ("B", 0.5)], parent=question1
    )
    question1_negative = EvidenceNode(
        "No", belief_state=[("C", 0.5), ("D", 0.5)], parent=question1
    )
    question1.children.extend([question1_affirmative, question1_negative])
    root.children.append(question1)

    question2 = QuestionNode("Is it an even letter?", parent=root)
    question2_affirmative = EvidenceNode(
        "Yes", belief_state=[("A", 0.5), ("C", 0.5)], parent=question2
    )
    question2_negative = EvidenceNode(
        "No", belief_state=[("B", 0.5), ("D", 0.5)], parent=question2
    )
    question2.children.extend([question2_affirmative, question2_negative])
    root.children.append(question2)

    print(stringify(root))
