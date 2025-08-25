from dataclasses import asdict

from method import LLMInteraction, RunHistory, UserInteraction
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


# TODO: Fix serialising
def serialise_run_history(history: RunHistory) -> dict:
    history_dict = asdict(history)

    history_dict["start_time"] = history.start_time.isoformat()
    if history.end_time:
        history_dict["end_time"] = history.end_time.isoformat()

    # Add a type field to each interaction
    processed_interactions = []
    for i, interaction in enumerate(history.interactions):
        interaction_dict = history_dict["interactions"][i]
        interaction_dict["timestamp"] = interaction.timestamp.isoformat()
        if isinstance(interaction, LLMInteraction):
            interaction_dict["type"] = "llm"
        elif isinstance(interaction, UserInteraction):
            interaction_dict["type"] = "user"
        processed_interactions.append(interaction_dict)
    history_dict["interactions"] = processed_interactions

    history_dict["tree_states"] = [serialise_tree(tree) for tree in history.tree_states]

    return history_dict


# TODO: Add deserialisation of run history
