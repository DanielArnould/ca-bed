"""
This module orchestrates the different methods of conversation employed
by enhanced UoT. It should handle the expansion and creation of the tree,
getting and processing the examiner response and the guesser response

**IMPORTANT**: As much info as possible should be saved with every run. This includes:
- Conversation history
- Tree state.
"""

from chat_utils import ask_llm

"""
TODO: Create method functions to 
orchestrate the tree creation and dialogue

Also handle open sets and final logic.

Also handle multiple benchmarks.
"""


def get_examiner_response(history: list[str]) -> str:
    msg = [history[0]] + history[-3:] if len(history) > 3 else history
    msg = "\n".join(msg)
    return ask_llm(msg)


def get_guesser_response(node: UoTNode) -> tuple[UoTNode, str]:
    if node.is_terminal():
        return node, node.items[0]


def converse(target: str) -> None:
    print(f"Target is {target}")
    print("------ DIALOGUE START ------")

    examiner_history = [prompts.examiner_prologue(target)]
    guesser_history = [prompts.guesser_prologue()]

    root = UoTNode()


def mcts(node: UoTNode) -> tuple[str, UoTNode, UoTNode]: ...


def get_user_response(question: str, history: list[str]) -> str: ...


def algorithm():
    history = []
    time = 0
    node = UoTNode()

    while time < 20 and len(node.items) > 1:
        question, yes_child, no_child = mcts(node)
        answer = get_user_response(question, history)
        node = yes_child if answer == "yes" else no_child
        history.extend((question, answer))
        time += 1

    # check if answer is correct
