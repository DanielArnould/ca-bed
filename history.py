from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from node import EvidenceNode, QuestionNode
from question_clustering import Cluster, QuestionClustering
from voyager import Index


@dataclass
class RunRecord:
    task_info: str
    true_answer: str
    start_time: datetime
    end_time: datetime
    total_input_tokens: int
    total_output_tokens: int
    final_path: list[str]
    final_answer: str
    serialised_tree: dict | None


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


def save_question_clustering(
    clustering: QuestionClustering, json_path: Path, voyager_path: Path
) -> None:
    serialised_clusters = {
        key: {
            "questions": cluster.questions,
            "likelihoods": cluster.likelihoods,
        }
        for key, cluster in clustering.clusters.items()
    }

    json_cluster = {"clusters": serialised_clusters, "threshold": clustering.threshold}

    with json_path.open("w") as f:
        json.dump(json_cluster, f)

    clustering.index.save(str(voyager_path))


def load_question_clustering(json_path: Path, voyager_path: Path) -> QuestionClustering:
    clustering = QuestionClustering(threshold=-1)

    with json_path.open("r") as f:
        qc_dict = json.load(f)

    clustering.threshold = qc_dict["threshold"]

    for key, cluster_dict in qc_dict["clusters"].items():
        cluster = Cluster(
            questions=cluster_dict["questions"],
            likelihoods=cluster_dict["likelihoods"],
        )
        clustering.clusters[key] = cluster

    with voyager_path.open("rb") as f:
        clustering.index = Index.load(f)

    return clustering


def serialise_run_record(history: RunRecord) -> dict:
    return {
        "task_info": history.task_info,
        "true_answer": history.true_answer,
        "start_time": history.start_time.isoformat(),
        "end_time": history.end_time.isoformat(),
        "total_input_tokens": history.total_input_tokens,
        "total_output_tokens": history.total_output_tokens,
        "final_path": history.final_path,
        "final_answer": history.final_answer,
        "tree": history.serialised_tree,
    }


def deserialise_run_record(
    history_dict: dict,
    include_tree: bool = False,
) -> RunRecord:
    return RunRecord(
        task_info=history_dict["task_info"],
        true_answer=history_dict["true_answer"],
        start_time=datetime.fromisoformat(history_dict["start_time"]),
        end_time=datetime.fromisoformat(history_dict["end_time"]),
        total_input_tokens=history_dict["total_input_tokens"],
        total_output_tokens=history_dict["total_output_tokens"],
        serialised_tree=history_dict["tree"] if include_tree else None,
        final_path=history_dict["final_path"],
        final_answer=history_dict["final_answer"],
    )
