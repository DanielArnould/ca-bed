from dataclasses import asdict
from datetime import datetime

from method import LLMInteraction, RunHistory, UserInteraction
from models import LLMOutput, Model, Token
from node import EvidenceNode, QuestionNode


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
        "start_time": history.start_time.isoformat(),
        "end_time": history.end_time.isoformat(),
        "final_path": history.final_path,
        "final_answer": history.final_answer,
    }

    # Add a type field to each interaction
    processed_interactions = []
    for interaction in history.interactions:
        match interaction:
            case LLMInteraction(timestamp, prompt, model, output):
                processed_interactions.append(
                    {
                        "type": "llm",
                        "timestamp": timestamp.isoformat(),
                        "prompt": prompt,
                        "model": model.name,
                        "output": asdict(output),
                    }
                )
            case UserInteraction(timestamp, question, options, selection):
                processed_interactions.append(
                    {
                        "type": "user",
                        "timestamp": timestamp.isoformat(),
                        "question": question,
                        "options": options,
                        "selection": selection,
                    }
                )

    history_dict["interactions"] = processed_interactions
    history_dict["tree_states"] = [serialise_tree(tree) for tree in history.tree_states]

    return history_dict


def deserialise_run_history(history_dict: dict) -> RunHistory:
    # Reconstruct interactions based on the 'type' field
    interactions = []
    for interaction_dict in history_dict["interactions"]:
        interaction_dict: dict
        interaction_type = interaction_dict["type"]
        timestamp = datetime.fromisoformat(interaction_dict["timestamp"])

        if interaction_type == "user":
            interactions.append(
                UserInteraction(
                    timestamp=timestamp,
                    question=interaction_dict["question"],
                    options=interaction_dict["options"],
                    selection=interaction_dict["selection"],
                )
            )
        elif interaction_type == "llm":
            model = Model[interaction_dict["model"]]
            output_data = interaction_dict["output"]
            tokens = (
                [Token(**t) for t in output_data["tokens"]]
                if output_data.get("tokens")
                else None
            )
            output = LLMOutput(
                string=output_data["string"],
                reasoning=output_data.get("reasoning"),
                tokens=tokens,
            )
            interactions.append(
                LLMInteraction(
                    timestamp=timestamp,
                    prompt=interaction_dict["prompt"],
                    model=model,
                    output=output,
                )
            )

    return RunHistory(
        task_info=history_dict["task_info"],
        start_time=datetime.fromisoformat(history_dict["start_time"]),
        end_time=datetime.fromisoformat(history_dict["end_time"]),
        interactions=interactions,
        tree_states=[
            deserialise_tree(tree_data) for tree_data in history_dict["tree_states"]
        ],
        final_path=history_dict["final_path"],
        final_answer=history_dict["final_answer"],
    )
