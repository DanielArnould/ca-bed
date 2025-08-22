"""
##################################################
#                   DEPRECATED                   #
##################################################

Only remains here as reference for designing the reward functions
for now
"""

from dataclasses import dataclass
from itertools import chain
import math
import random

from chat_utils import create_questions

"""
TODO: Extract MCTS logic in nodes to separate functions?
"""


@dataclass(slots=True)
class NodePair:
    question: str
    affirmative: "Node"
    negative: "Node"
    parent: "Node"

    def immediate_reward(self) -> float:
        """Information gain at current node pair"""
        lambda_scale = 0.4

        affirmative_size = len(self.affirmative.items)
        negative_size = len(self.negative.items)

        proportion_affirmative = affirmative_size / (affirmative_size + negative_size)
        proportion_negative = negative_size / (affirmative_size + negative_size)

        unscaled_gain = -proportion_affirmative * math.log2(
            proportion_affirmative
        ) - proportion_negative * math.log(proportion_negative)

        # Scaling term is added for better empircal results
        scaled_gain = unscaled_gain / (
            1 + (1 / lambda_scale) * abs(proportion_affirmative - proportion_negative)
        )
        return scaled_gain

    def accumulated_reward(self) -> float:
        """Total information gained of the interaction trajectory"""
        # Safe to ignore typing since the only case where the parent does not have a pair is if the parent is a root node
        return self.immediate_reward() + (
            self.parent.pair.immediate_reward()  # type: ignore
            if isinstance(self.parent, "Node")
            else 0
        )

    def expected_reward(self) -> float:
        """Expected reward of asking a question"""
        affirmative_size = len(self.affirmative.items)
        negative_size = len(self.negative.items)

        proportion_affirmative = affirmative_size / (affirmative_size + negative_size)
        proportion_negative = negative_size / (affirmative_size + negative_size)

        return (
            proportion_affirmative * self.affirmative.expected_reward()
            + proportion_negative * self.negative.expected_reward()
        )


class Node:
    items: list[str]
    pair: NodePair | None
    children: list[NodePair]
    cumulative_reward: float = 0

    _number_of_visits: int

    def __init__(self, items: list[str]): ...

    def expected_reward(self) -> float:
        """Expected reward of asking a question and the answer being this node"""
        assert self.pair is not None, "Cannot calculate expected reward of root!"

        if not self.children:
            return self.pair.immediate_reward()

        return (
            1
            / (len(self.children))
            * sum(pair.expected_reward() for pair in self.children)
        )

    def best_question(self) -> NodePair:
        """Select the best question for future interactions from this node. Designed for use after MCTS"""
        assert len(self.children) > 0, "No questions to pick from!"
        return max(self.children, key=lambda pair: pair.expected_reward())

    def select(self) -> "Node":
        """Select a node for exploration in MCTS"""
        exploration_constant = 0.1
        parent_visit_count = self._number_of_visits

        def uct(node: "Node") -> float:
            return node.cumulative_reward / node._number_of_visits + (
                exploration_constant
                * (math.log2(parent_visit_count / node._number_of_visits))
            )

        return max(
            (
                child
                for child in chain.from_iterable(
                    (child.affirmative, child.negative) for child in self.children
                )
            ),
            key=uct,
        )

    def expand(self) -> None:
        """Create a new layer of children (affirmative + negative node splits) from the current node"""
        previous_questions = []  # TODO: Backtrack up the tree to get previous questions
        n = 3  # TODO: Why does the paper not specify the branching factor of questions?

        questions = create_questions(self.items, previous_questions, 3)

        pairs = [
            NodePair(
                question.question,
                Node(question.items_yes),
                Node(question.items_no),
                self,
            )
            for question in questions
        ]

        # TODO: Can we remove this cyclic link?
        for pair in pairs:
            pair.affirmative.pair = pair
            pair.negative.pair = pair

        self.children.extend(pairs)

    def is_terminal_node(self) -> bool:
        # TODO: The paper just asks direct questions at this point
        # But those don't end up creating nodes. Could we define strictly less
        # than 2 items to be terminal?
        return len(self.items) <= 2

    @staticmethod
    def rollout(curr: "Node") -> "Node":
        """Simulate a random interaction up to a predefined depth or terminal state"""
        max_depth = 6

        for _ in range(max_depth):
            if not curr.children:
                # " following a single level of expansion if child nodes did not exist"
                # TODO: Check if we should not expand again if we already expanded once.
                curr.expand()

            available_children = list(
                chain.from_iterable(
                    (pair.affirmative, pair.negative) for pair in curr.children
                )
            )
            next_node = Node.rollout_policy(available_children)
            curr = next_node

        # TODO: Could we just return the expected reward?
        return curr

    @staticmethod
    def rollout_policy(possible_children: list["Node"]) -> "Node":
        return random.choice(possible_children)

    @staticmethod
    def backpropagate(start_of_rollout: "Node", end_of_rollout: "Node") -> None:
        curr = start_of_rollout
        while True:
            curr.cumulative_reward += curr.expected_reward()

            if curr == end_of_rollout:
                break

            assert curr.pair is not None, "Node does not not belong to a pair!"
            curr = curr.pair.parent

    @staticmethod
    def mcts(node: "Node") -> None:
        K_ITERATIONS = 10  # TODO: FA-MCTS paper uses 10 iterations in experiments. Why? Can we identify an optimal?

        for _ in range(K_ITERATIONS):
            if not node.children:
                node.expand()
            selected_child = node.select()
            end_of_rollout = Node.rollout(selected_child)
            Node.backpropagate(node, end_of_rollout)
