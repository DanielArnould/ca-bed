import math

from node import QuestionNode


def shannon_entropy(belief_state: dict[str, float]) -> float:
    return -sum(prob * math.log2(prob) for prob in belief_state.values() if prob > 0)


def information_gain(question: QuestionNode) -> float:
    assert len(question.children) > 0, "Question has no simulated answers!"

    parent_entropy = shannon_entropy(question.parent.belief_state)
    weighted_average_entropy = sum(
        evidence.marginal_likelihood * shannon_entropy(evidence.belief_state)
        for evidence in question.children
    )

    return parent_entropy - weighted_average_entropy


def immediate_reward(question: QuestionNode, penalty_scalar: float = 0.4) -> float:
    max_likelihood = max(evidence.marginal_likelihood for evidence in question.children)
    min_likelihood = min(evidence.marginal_likelihood for evidence in question.children)
    specificity_penalty = penalty_scalar * max_likelihood - min_likelihood
    return information_gain(question) / (1 + penalty_scalar * specificity_penalty)


def accumulated_reward(question: QuestionNode) -> float:
    return immediate_reward(question) + (
        accumulated_reward(question.parent.parent)
        if question.parent.parent is not None
        else 0
    )


def expected_future_reward(question: QuestionNode) -> float:
    is_leaf_question = not any(
        len(evidence.children) > 0 for evidence in question.children
    )
    if is_leaf_question:
        return accumulated_reward(question)

    expected_reward = 0.0
    for evidence in question.children:
        s = sum(expected_future_reward(q) for q in evidence.children) / len(
            evidence.children
        )
        expected_reward += evidence.marginal_likelihood * s

    return sum(
        (
            sum(
                expected_future_reward(future_question)
                for future_question in evidence.children
            )
            / len(evidence.children)
        )
        * evidence.marginal_likelihood
        for evidence in question.children
    )
