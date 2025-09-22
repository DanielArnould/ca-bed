from datetime import datetime
import logging
from history import RunRecord, serialise_tree
from models import Model, call_llm
from node import EvidenceNode, QuestionNode, get_conversation_depth
from tasks.direct_prompting_task import DirectPromptingTask, NaiveQuestionerResponse


LOGGER = logging.getLogger("Direct Prompting Method")


class DirectPromptingMethod:
    """
    The direct prompting method fits into the same evaluation framework as the
    normal method. We achieve this as follows:

    1. We construct the tree as a straight path dynamically
    2. Whenever we make a prediction, we create a question node with the
       question 'Is it {hypothesis}?'
    3. Belief state is initially empty, but for every prediction we increase
       the hypothesis by 1/prediction_count and then normalise. This means if we
       keep making the same prediction, we will reflect that in the belief
       state, and the belief state will represent the order in which we make
       predictions. This makes top1 and top3 metrics work appropriately.
    4. For regular questions, the belief state remains unchanged
    5. We terminate whenever the task answer has a belief greater than 0
       (has been predicted at least once) or we reach max conversation depth
    """

    benchmark_model: Model
    method_model: Model
    task: DirectPromptingTask
    root: EvidenceNode
    current_node: EvidenceNode
    total_input_tokens: int
    total_output_tokens: int
    prediction_count: int

    def __init__(
        self,
        benchmark_model: Model,
        method_model: Model,
        task: DirectPromptingTask,
    ):
        self.benchmark_model = benchmark_model
        self.method_model = method_model
        self.task = task
        self.root = EvidenceNode(
            answer="ROOT",
            belief_state={},
            marginal_likelihood=1.0,
        )
        self.current_node = self.root
        self.total_input_tokens = self.total_output_tokens = 0
        self.prediction_count = 0

    async def run(self) -> RunRecord:
        start_time = datetime.now()
        final_path = [str(self.current_node)]

        while not self.is_terminal(self.current_node):
            # Get response from questioner model
            questioner_output, input_tokens, output_tokens = await call_llm(
                self.task.get_questioner_prompt(self.current_node), self.method_model
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            response_type, content = self.task.parse_questioner_output(
                questioner_output
            )

            match response_type:
                case NaiveQuestionerResponse.PREDICTION:
                    prediction = content
                    # Create question node for prediction
                    question_node = QuestionNode(
                        f"Is it {prediction}?", parent=self.current_node
                    )
                    self.current_node.children.append(question_node)
                    final_path.append(str(question_node))

                    updated_belief_state = self.calculate_posterior(
                        self.current_node.belief_state, prediction
                    )
                    # Get answer deterministically by comparing to expected answer
                    evidence_answer = (
                        "Yes"
                        if prediction.strip().lower()
                        == self.task.task_answer.strip().lower()
                        else "No"
                    )

                case NaiveQuestionerResponse.QUESTION:
                    question = content
                    # Create question node for regular question
                    question_node = QuestionNode(question, parent=self.current_node)
                    self.current_node.children.append(question_node)
                    final_path.append(str(question_node))

                    # Get answer from benchmark model
                    answerer_output, input_tokens, output_tokens = await call_llm(
                        self.task.get_answerer_prompt(question),
                        self.benchmark_model,
                    )
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens

                    # Belief state unchanged for regular questions
                    updated_belief_state = self.current_node.belief_state.copy()
                    evidence_answer = answerer_output

            evidence_node = EvidenceNode(
                answer=evidence_answer,
                belief_state=updated_belief_state,
                marginal_likelihood=1.0,
                parent=question_node,
            )
            question_node.children.append(evidence_node)
            self.current_node = evidence_node
            final_path.append(str(evidence_node))

        end_time = datetime.now()
        LOGGER.info(f"Completed run in {end_time - start_time}s!")

        return RunRecord(
            task_info=str(self.task),
            true_answer=self.task.task_answer,
            start_time=start_time,
            end_time=end_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            serialised_tree=serialise_tree(self.root),
            final_path=final_path,
            final_belief_state=self.current_node.belief_state,
        )

    def calculate_posterior(
        self, prior_belief_state: dict[str, float], prediction: str
    ) -> dict[str, float]:
        """Update belief state by adding weighted prediction and normalising"""
        self.prediction_count += 1

        # Earlier predictions get higher weight
        prediction_weight = 1.0 / self.prediction_count

        posterior = prior_belief_state.copy()
        posterior[prediction] += posterior.get(prediction, 0) + prediction_weight

        # Normalise to ensure probabilities sum to 1
        total_probability = sum(posterior.values())
        assert total_probability > 0
        posterior = {
            hypothesis: probability / total_probability
            for hypothesis, probability in posterior.items()
        }
        return posterior

    def is_terminal(self, node: EvidenceNode) -> bool:
        return (
            get_conversation_depth(node) >= self.task.max_conversation_depth
            or self.task.task_answer in node.belief_state
        )


def get_uniform_belief_state(hypothesis_space: list[str]) -> dict[str, float]:
    uniform_probability = 1.0 / len(hypothesis_space)
    return {hypothesis: uniform_probability for hypothesis in hypothesis_space}
