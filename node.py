"""
This module contains the key constructs for the tree structure needed for enhanced UoT.
The tree structure ought to be as extendable as possible, but reward functions
may need to be differentiated and selected by the method.
"""

from dataclasses import dataclass, field


@dataclass
class EvidenceNode:
    answer: str
    belief_state: dict[str, float]
    marginal_likelihood: float  # probability of picking this answer
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
            case EvidenceNode(answer, belief_state, marginal_likelihood, _, children):
                formatted_belief_state = ", ".join(
                    [
                        f"{hypothesis}: {prob}"
                        for hypothesis, prob in belief_state.items()
                    ]
                )
                lines.append(
                    f"{prefix}{connector}Answer: {answer} | Marginal Likelihood: {marginal_likelihood} | Belief State: [{formatted_belief_state}]"
                )

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)
            case QuestionNode(question, _, children):
                lines.append(f"{prefix}{connector}Question: {question}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)

    formatted_belief_state = ", ".join(
        [f"{hypothesis}: {prob}" for hypothesis, prob in root.belief_state.items()]
    )
    lines.append(
        f"Answer: {root.answer} | Marginal Likelihood: {root.marginal_likelihood} | Belief State: [{formatted_belief_state}]"
    )
    for i, child in enumerate(root.children):
        _build_string(child, "", i == len(root.children) - 1)

    return "\n".join(lines)


if __name__ == "__main__":
    root = EvidenceNode(
        "ROOT",
        belief_state=dict([("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)]),
        marginal_likelihood=1.0,
    )

    question1 = QuestionNode("Is it greater than or equal to B?", parent=root)
    question1_affirmative = EvidenceNode(
        "Yes",
        belief_state=dict([("A", 0.5), ("B", 0.5)]),
        marginal_likelihood=0.5,
        parent=question1,
    )
    question1_negative = EvidenceNode(
        "No",
        belief_state=dict([("C", 0.5), ("D", 0.5)]),
        marginal_likelihood=0.5,
        parent=question1,
    )
    question1.children.extend([question1_affirmative, question1_negative])
    root.children.append(question1)

    question2 = QuestionNode("Is it an even letter?", parent=root)
    question2_affirmative = EvidenceNode(
        "Yes",
        belief_state=dict([("A", 0.5), ("C", 0.5)]),
        marginal_likelihood=0.5,
        parent=question2,
    )
    question2_negative = EvidenceNode(
        "No",
        belief_state=dict([("B", 0.5), ("D", 0.5)]),
        marginal_likelihood=0.5,
        parent=question2,
    )
    question2.children.extend([question2_affirmative, question2_negative])
    root.children.append(question2)

    print(stringify(root))
