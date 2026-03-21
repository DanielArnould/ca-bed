from dataclasses import dataclass, field

type ProbabilityDistribution = dict[str, float]
type Likelihoods = dict[str, float]


@dataclass(slots=True, frozen=True)
class EvidenceNode:
    answer: str
    belief_state: ProbabilityDistribution
    marginal_likelihood: float
    parent: "QuestionNode | None" = None
    children: list["QuestionNode"] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Answer: '{self.answer}' | Marginal Likelihood: {self.marginal_likelihood} | Belief State: {self.belief_state}"


@dataclass(slots=True, frozen=True)
class QuestionNode:
    question: str
    possible_answers: list[str]
    parent: "EvidenceNode"
    children: list["EvidenceNode"] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Question: '{self.question}' | Possible Answers: {self.possible_answers}"
        )


@dataclass(slots=True, frozen=True)
class QuestionAnswer:
    question: str
    answer: str


def get_conversation_depth(node: EvidenceNode) -> int:
    return len(get_conversation_history(node))


def get_conversation_history(node: EvidenceNode) -> list[QuestionAnswer]:
    history = []
    curr = node
    while curr.parent:
        question = curr.parent.question
        answer = curr.answer
        history.append(QuestionAnswer(question, answer))
        curr = curr.parent.parent

    return history[::-1]


def pprint(root: EvidenceNode) -> None:
    lines = []

    def _build_string(
        node: EvidenceNode | QuestionNode, prefix: str = "", is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "

        match node:
            case EvidenceNode(_, _, _, _, children):
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)

            case QuestionNode(_, _, _, children):
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)

    lines.append(str(root))
    for i, child in enumerate(root.children):
        _build_string(child, "", i == len(root.children) - 1)

    print("\n".join(lines))
