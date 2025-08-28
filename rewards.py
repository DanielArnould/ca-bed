import math

from node import EvidenceNode, QuestionNode


def shannon_entropy(belief_state: dict[str, float]) -> float:
    return -sum(prob * math.log2(prob) for prob in belief_state.values() if prob > 0)


def information_gain(
    prior_belief_state: dict[str, float], posterior_belief_state: dict[str, float]
) -> float:
    return shannon_entropy(prior_belief_state) - shannon_entropy(posterior_belief_state)


# Lambda is set to 0.4 in UoT Paper (see page 20, I.6)
def specificity_penalty(question: QuestionNode, penalty_scalar: float = 0.4) -> float:
    max_likelihood = max(evidence.marginal_likelihood for evidence in question.children)
    min_likelihood = min(evidence.marginal_likelihood for evidence in question.children)
    return penalty_scalar * max_likelihood - min_likelihood


def immediate_reward(evidence: EvidenceNode) -> float:
    assert evidence.parent is not None, "Cannot determine reward of root node!"
    return information_gain(
        evidence.parent.parent.belief_state, evidence.belief_state
    ) / (1 + specificity_penalty(evidence.parent))


def accumulated_reward(evidence: EvidenceNode) -> float:
    if evidence.parent is None:
        return 0

    return immediate_reward(evidence) + accumulated_reward(evidence.parent.parent)


def expected_reward(question: QuestionNode) -> float:
    assert len(question.children) > 0, "Question has no answers!"

    weighted_mean_reward = 0.0
    for evidence in question.children:
        evidence_reward = (
            sum(
                expected_reward(future_question)
                for future_question in evidence.children
            )
            / len(evidence.children)
            if len(evidence.children) > 0
            else accumulated_reward(evidence)
        )
        weighted_mean_reward += evidence.marginal_likelihood * evidence_reward

    return weighted_mean_reward
