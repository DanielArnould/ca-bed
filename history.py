from dataclasses import dataclass
from datetime import datetime

from node import EvidenceNode, QuestionNode


@dataclass
class RunHistory:
    task_info: str
    actual_answer: str
    start_time: datetime
    end_time: datetime
    # Deep copies of the root node at each iteration
    tree_states: list[EvidenceNode]
    final_path: list[str]
    final_answer: str
    question_clusters: list[list[str]]


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
                assert parent is None or isinstance(parent, QuestionNode), (
                    "Invalid parent of evidence node!"
                )
                node = EvidenceNode(
                    answer=data["answer"],
                    belief_state=data["belief_state"],
                    marginal_likelihood=data["marginal_likelihood"],
                    parent=parent,
                )
            case "question":
                assert isinstance(parent, EvidenceNode), (
                    "Invalid parent of question node!"
                )
                node = QuestionNode(question=data["question"], parent=parent)
            case node_type:
                raise ValueError(f"Unknown node type: {node_type}")

        node.children = [  # type: ignore
            _deserialise(child_data, parent=node) for child_data in data["children"]
        ]
        return node

    return _deserialise(serialised_tree, parent=None)  # type: ignore


def serialise_run_history(history: RunHistory) -> dict:
    history_dict = {
        "task_info": history.task_info,
        "actual_answer": history.actual_answer,
        "start_time": history.start_time.isoformat(),
        "end_time": history.end_time.isoformat(),
        "final_path": history.final_path,
        "final_answer": history.final_answer,
        "question_clusters": history.question_clusters,
    }

    history_dict["tree_states"] = [serialise_tree(tree) for tree in history.tree_states]

    return history_dict


def deserialise_run_history(history_dict: dict) -> RunHistory:
    return RunHistory(
        task_info=history_dict["task_info"],
        actual_answer=history_dict["actual_answer"],
        start_time=datetime.fromisoformat(history_dict["start_time"]),
        end_time=datetime.fromisoformat(history_dict["end_time"]),
        tree_states=[
            deserialise_tree(tree_data) for tree_data in history_dict["tree_states"]
        ],
        final_path=history_dict["final_path"],
        final_answer=history_dict["final_answer"],
        question_clusters=history_dict["question_clusters"],
    )
