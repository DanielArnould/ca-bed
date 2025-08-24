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

    def __str__(self) -> str:
        formatted_belief_state = ", ".join(
            [f"{hypothesis}: {prob}" for hypothesis, prob in self.belief_state.items()]
        )
        return f"Answer: {self.answer} | Marginal Likelihood: {self.marginal_likelihood} | Belief State: [{formatted_belief_state}]"


@dataclass
class QuestionNode:
    question: str
    parent: EvidenceNode
    children: list[EvidenceNode] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Question: {self.question}"


def serialise_tree(root: EvidenceNode) -> dict:
    def _serialise(node: EvidenceNode | QuestionNode) -> dict:
        match node:
            case EvidenceNode(answer, belief_state, marginal_likelihood, _, children):
                return {
                    "type": "evidence",
                    "answer": answer,
                    "belief_state": belief_state,
                    "marginal_likelihood": marginal_likelihood,
                    "children": [_serialise(child) for child in children],
                }
            case QuestionNode(question, _, children):
                return {
                    "type": "question",
                    "question": question,
                    "children": [_serialise(child) for child in children],
                }

    return _serialise(root)


def deserialise_tree(serialised_tree: dict) -> EvidenceNode:
    def _deserialise(
        data: dict, parent: EvidenceNode | QuestionNode | None = None
    ) -> EvidenceNode | QuestionNode:
        match data["type"]:
            case "evidence":
                assert parent is None or isinstance(parent, QuestionNode)
                node = EvidenceNode(
                    answer=data["answer"],
                    belief_state=data["belief_state"],
                    marginal_likelihood=data["marginal_likelihood"],
                    parent=parent,
                )
            case "question":
                assert isinstance(parent, EvidenceNode)
                node = QuestionNode(question=data["question"], parent=parent)
            case node_type:
                raise ValueError(f"Unknown node type: {node_type}")

        node.children = [  # type: ignore
            _deserialise(child_data, parent=node) for child_data in data["children"]
        ]
        return node

    return _deserialise(serialised_tree, parent=None)  # type: ignore


def stringify(root: EvidenceNode) -> str:
    lines = []

    def _build_string(
        node: EvidenceNode | QuestionNode, prefix: str = "", is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "

        match node:
            case EvidenceNode(_, _, _, _, children) as node:
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)
            case QuestionNode(_, _, children) as node:
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)

    lines.append(str(root))
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
    serialised_tree = serialise_tree(root)
    print(serialised_tree)
    deserialised_tree = deserialise_tree(serialised_tree)
    print(stringify(deserialised_tree))
