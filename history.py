from dataclasses import dataclass
from datetime import datetime

import torch

from node import EvidenceNode, QuestionNode
from question_clustering import Cluster, QuestionClustering


@dataclass
class RunHistory:
    task_info: str
    actual_answer: str
    start_time: datetime
    end_time: datetime
    tree: EvidenceNode
    final_path: list[str]
    final_answer: str
    question_clustering: QuestionClustering


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


def serialise_question_clustering(clustering: QuestionClustering) -> dict:
    serialised_clusters = [
        {
            "centroid": cluster.centroid.cpu().numpy().tolist(),
            "questions": cluster.questions,
            "likelihoods": cluster.likelihoods,
        }
        for cluster in clustering.clusters
    ]

    return {
        "model_name": clustering.model_name,
        "threshold": clustering.threshold,
        "clusters": serialised_clusters,
    }


def deserialise_question_clustering(qc_dict: dict) -> QuestionClustering:
    clustering = QuestionClustering(
        threshold=qc_dict["threshold"], model_name=qc_dict["model_name"]
    )

    for cluster_dict in qc_dict["clusters"]:
        centroid = torch.tensor(cluster_dict["centroid"], dtype=torch.float32)

        cluster = Cluster(
            centroid=centroid,
            questions=[cluster_dict["questions"]],
            likelihoods=cluster_dict["likelihoods"],
        )
        clustering.clusters.append(cluster)

    return clustering


def serialise_run_history(history: RunHistory) -> dict:
    return {
        "task_info": history.task_info,
        "actual_answer": history.actual_answer,
        "start_time": history.start_time.isoformat(),
        "end_time": history.end_time.isoformat(),
        "final_path": history.final_path,
        "final_answer": history.final_answer,
        "question_clusters": serialise_question_clustering(history.question_clustering),
        "tree": serialise_tree(history.tree),
    }


def deserialise_run_history(history_dict: dict) -> RunHistory:
    return RunHistory(
        task_info=history_dict["task_info"],
        actual_answer=history_dict["actual_answer"],
        start_time=datetime.fromisoformat(history_dict["start_time"]),
        end_time=datetime.fromisoformat(history_dict["end_time"]),
        tree=deserialise_tree(history_dict["tree"]),
        final_path=history_dict["final_path"],
        final_answer=history_dict["final_answer"],
        question_clustering=deserialise_question_clustering(
            history_dict["question_clusters"]
        ),
    )
