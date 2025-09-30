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


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


def get_conversation_history(node: EvidenceNode) -> list[tuple[str, str]]:
    history = []
    curr = node
    while curr.parent:
        question = curr.parent.question
        answer = curr.answer
        history.append((question, answer))
        curr = curr.parent.parent

    return history[::-1]


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
