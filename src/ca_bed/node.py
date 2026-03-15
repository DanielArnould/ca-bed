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


@dataclass(slots=True, frozen=True)
class QuestionNode:
    question: str
    possible_answers: list[str]
    parent: "EvidenceNode"
    children: list["EvidenceNode"] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class QuestionAnswer:
    question: str
    answer: str


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


def get_conversation_history(node: EvidenceNode) -> list[QuestionAnswer]:
    history = []
    curr = node
    while curr.parent:
        question = curr.parent.question
        answer = curr.answer
        history.append(QuestionAnswer(question, answer))
        curr = curr.parent.parent

    return history[::-1]
