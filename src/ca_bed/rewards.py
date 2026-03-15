import math

from src.ca_bed.node import EvidenceNode, ProbabilityDistribution, QuestionNode


def shannon_entropy(belief_state: ProbabilityDistribution) -> float:
    return -sum(prob * math.log2(prob) for prob in belief_state.values() if prob > 0)


def information_gain(
    prior_belief_state: ProbabilityDistribution,
    posterior_belief_state: ProbabilityDistribution,
) -> float:
    return shannon_entropy(prior_belief_state) - shannon_entropy(posterior_belief_state)


def reward(node: EvidenceNode | QuestionNode) -> float:
    match node:
        case QuestionNode():
            return sum(
                evidence.marginal_likelihood * reward(evidence)
                for evidence in node.children
            )

        case EvidenceNode():
            # If we can ask more questions, pick the absolute best one
            if node.children:
                return max(reward(child) for child in node.children)

            # Base case (leaf node): Calculate actual information gain
            assert node.parent is not None, (
                f"{node} does not have a parent and hence a reward is not defined"
            )
            return information_gain(node.parent.parent.belief_state, node.belief_state)
