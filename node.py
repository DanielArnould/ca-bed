"""
This module contains the key constructs for the tree structure needed for enhanced UoT.
The tree structure ought to be as extendable as possible, but reward functions
may need to be differentiated and selected by the method.
"""

from dataclasses import dataclass, field

type Probability = float
type Item = str


@dataclass
class AnswerNode:
    answer: str
    item_probabilities: list[tuple[Item, Probability]]
    this_group: "QuestionGroup | None" = None
    children_groups: list["QuestionGroup"] = field(default_factory=list)


@dataclass
class QuestionGroup:
    question: str
    parent: AnswerNode
    answer_nodes: list[AnswerNode] = field(default_factory=list)


def stringify(root: AnswerNode) -> str:
    lines = []

    def _build_string(
        node: AnswerNode | QuestionGroup, prefix: str = "", is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "

        match node:
            case AnswerNode(answer, item_probabilities, _, children_groups):
                formatted_item_probabilities = ", ".join(
                    [f"{item}: {prob}" for item, prob in item_probabilities]
                )
                lines.append(
                    f"{prefix}{connector}Answer: {answer} | Probabilities: [{formatted_item_probabilities}]"
                )

                for i, child in enumerate(children_groups):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children_groups) - 1)
            case QuestionGroup(question, _, answer_nodes):
                lines.append(f"{prefix}{connector}Question: {question}")

                for i, child in enumerate(answer_nodes):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(answer_nodes) - 1)

    formatted_item_probabilities = ", ".join(
        [f"{item}: {prob}" for item, prob in root.item_probabilities]
    )
    lines.append(
        f"Answer: {root.answer} | Probabilities: [{formatted_item_probabilities}]"
    )
    for i, child in enumerate(root.children_groups):
        _build_string(child, "", i == len(root.children_groups) - 1)

    return "\n".join(lines)


if __name__ == "__main__":
    root = AnswerNode(
        "ROOT", item_probabilities=[("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)]
    )

    question1 = QuestionGroup("Is it greater than or equal to B?", parent=root)
    question1_affirmative = AnswerNode(
        "Yes", item_probabilities=[("A", 0.5), ("B", 0.5)], this_group=question1
    )
    question1_negative = AnswerNode(
        "No", item_probabilities=[("C", 0.5), ("D", 0.5)], this_group=question1
    )
    question1.answer_nodes.extend([question1_affirmative, question1_negative])
    root.children_groups.append(question1)

    question2 = QuestionGroup("Is it an even letter?", parent=root)
    question2_affirmative = AnswerNode(
        "Yes", item_probabilities=[("A", 0.5), ("C", 0.5)], this_group=question2
    )
    question2_negative = AnswerNode(
        "No", item_probabilities=[("B", 0.5), ("D", 0.5)], this_group=question2
    )
    question2.answer_nodes.extend([question2_affirmative, question2_negative])
    root.children_groups.append(question2)

    print(stringify(root))
