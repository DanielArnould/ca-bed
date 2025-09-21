import asyncio
from datetime import datetime
from functools import partial
import logging
from typing_extensions import deprecated
from history import RunRecord, serialise_tree
from models import Model, call_llm
from node import EvidenceNode, QuestionNode
from question_clustering import QuestionClustering
from rewards import expected_reward
from tasks.task import Task, DPTask

LOGGER = logging.getLogger("Method")

class DPMethod:
    benchmark_model: Model
    method_model: Model
    task: DPTask
    root: EvidenceNode
    total_input_tokens: int
    total_output_tokens: int
    has_questioner_predicted: bool
    perseverance: bool
    history: list[tuple[str, str]]

    def __init__(
        self,
        benchmark_model: Model,
        method_model: Model,
        task: DPTask,
        perseverance: bool = False,
    ):
        self.benchmark_model = benchmark_model
        self.method_model = method_model
        self.task = task
        self.perseverance = perseverance
        self.root = EvidenceNode(
            answer="ROOT",
            belief_state={},
            marginal_likelihood=1.0,
        )
        self.total_input_tokens = self.total_output_tokens = 0
        self.history = []


    async def run(self) -> RunRecord:
        LOGGER.info('Starting Direct Prompting Run..')
        start_time = datetime.now()
        final_path = []

        final_answer: str | None = None

        final_path.append(f"START")

        for i in range(self.task.max_conversation_depth):
            questioner_output, input_tokens, output_tokens = await call_llm(
                self.task.get_questioner_prompt(self.history), self.method_model
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            has_questioner_predicted, answer_or_question = self.task.parse_questioner_output(questioner_output)

            if has_questioner_predicted:
                prediction = answer_or_question
                if self.perseverance and prediction.strip().lower() != self.task.task_answer.strip().lower():
                    LOGGER.info(f"Incorrect prediction at turn {i+1}: {prediction}. Continuing due to perseverance.")
                    final_path.append(f"INCORRECT PREDICTION: {prediction}")
                    continue
                else:
                    final_answer = prediction
                    final_path.append(f"PREDICTION: {final_answer}")
                    break

            question = answer_or_question
            final_path.append(f"Q: {question}")

            answerer_output, input_tokens, output_tokens = await call_llm(
                self.task.get_answerer_prompt(question), self.benchmark_model
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            answer = self.task.parse_answerer_output(answerer_output)
            self.history.append((question, answer))
            final_path.append(f"A: {answer}")

        end_time = datetime.now()
        LOGGER.info(
            f"Completed run in {end_time - start_time}s! Final answer: {final_answer}"
        )

        # Create a mock belief state for the RunRecord
        final_belief_state = {final_answer: 1.0} if final_answer else {}


        return RunRecord(
            task_info=str(self.task),
            true_answer=self.task.task_answer,
            start_time=start_time,
            end_time=end_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            serialised_tree=serialise_tree(self.root),
            final_path=final_path,
            final_belief_state=final_belief_state,
        )


class Method:
    benchmark_model: Model
    method_model: Model
    sharpness_constant: float
    task: Task
    question_clustering: QuestionClustering
    root: EvidenceNode
    total_input_tokens: int
    total_output_tokens: int

    def __init__(
        self,
        benchmark_model: Model,
        method_model: Model,
        sharpness_constant: float,
        task: Task,
        question_clustering: QuestionClustering,
    ):
        self.benchmark_model = benchmark_model
        self.method_model = method_model
        self.sharpness_constant = sharpness_constant
        self.task = task
        self.question_clustering = question_clustering
        self.root = EvidenceNode(
            answer="ROOT",
            belief_state={},
            marginal_likelihood=1.0,
        )
        self.total_input_tokens = self.total_output_tokens = 0

    async def run(self) -> RunRecord:
        start_time = datetime.now()
        final_path = []

        prior_prompt = self.task.get_prior_prompt()
        prior_output, input_tokens, output_tokens = await call_llm(
            prior_prompt, self.method_model
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.root.belief_state = self.task.parse_prior_output(prior_output)
        LOGGER.info(f"Created initial belief state: {str(self.root)}")

        current_node = self.root
        final_path.append(str(current_node))

        while not self.is_terminal(current_node):
            await self.expand_evidence(current_node, 0)
            best_question_node = max(
                current_node.children,
                key=partial(
                    expected_reward, sharpness_constant=self.sharpness_constant
                ),
            )
            LOGGER.info(f"Selected question node: {str(best_question_node)}")
            final_path.append(str(best_question_node))

            selection_prompt = self.task.get_answer_selection_prompt(best_question_node)
            LOGGER.info("Posing question to Benchmark LLM...")
            selection_output, input_tokens, output_tokens = await call_llm(
                selection_prompt, self.benchmark_model
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            selected_evidence_node = self.task.parse_answer_selection_output(
                selection_output, best_question_node
            )
            LOGGER.info(f"Benchmark LLM selected: {str(selected_evidence_node)}")
            current_node = selected_evidence_node

            final_path.append(str(current_node))

        end_time = datetime.now()
        LOGGER.info(
            f"Completed run in {end_time - start_time}s! Final belief: {current_node.belief_state}"
        )

        return RunRecord(
            task_info=str(self.task),
            true_answer=self.task.task_answer,
            start_time=start_time,
            end_time=end_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            serialised_tree=serialise_tree(self.root),
            final_path=final_path,
            final_belief_state=current_node.belief_state,
        )

    async def expand_evidence(self, curr: EvidenceNode, current_depth: int) -> None:
        if self.is_terminal(curr) or current_depth >= self.task.max_lookahead_depth:
            return

        if not curr.children:
            prompt = self.task.get_question_generation_prompt(curr)
            output, input_tokens, output_tokens = await call_llm(
                prompt, self.method_model
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            new_questions = self.task.parse_question_generation_output(output)
            new_question_nodes = [QuestionNode(q, curr) for q in new_questions]
            curr.children.extend(new_question_nodes)

        await asyncio.gather(
            *[self.expand_questions(child, current_depth) for child in curr.children]
        )

    async def expand_questions(self, curr: QuestionNode, current_depth: int) -> None:
        if not curr.children:
            cluster = self.question_clustering.get_cluster(curr.question)

            async with cluster.lock:
                if cluster.likelihoods is None:
                    output, input_tokens, output_tokens = await call_llm(
                        self.task.get_likelihood_elicitation_prompt(curr.question),
                        self.method_model,
                    )
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    cluster.likelihoods = self.task.parse_likelihood_elicitation_output(
                        output
                    )

            for answer, likelihoods in cluster.likelihoods.items():
                posterior, marginal = self.calculate_posterior(
                    curr.parent.belief_state, likelihoods
                )
                evidence_node = EvidenceNode(
                    answer=answer,
                    belief_state=posterior,
                    marginal_likelihood=marginal,
                    parent=curr,
                )
                curr.children.append(evidence_node)

        await asyncio.gather(
            *[self.expand_evidence(child, current_depth + 1) for child in curr.children]
        )

    def calculate_posterior(
        self, prior: dict[str, float], likelihoods: dict[str, float]
    ) -> tuple[dict[str, float], float]:
        # Calculate unnormalised posterior: P(hypothesis) * P(evidence|hypothesis)
        unnormalised_posterior: dict[str, float] = {}
        for hypo, prior_belief in prior.items():
            likelihood = likelihoods.get(hypo, 1.0)
            if hypo not in likelihoods:
                LOGGER.warning(f"{hypo} not found in likelihoods! Defaulting to 1...")
            unnormalised_posterior[hypo] = max(prior_belief * likelihood, 1e-10)

        # Calculate marginal (normalisation constant)
        marginal = sum(unnormalised_posterior.values())
        posterior = {
            hypo: (prob / marginal) for hypo, prob in unnormalised_posterior.items()
        }

        return posterior, marginal

    def is_terminal(self, node: EvidenceNode) -> bool:
        return get_conversation_depth(node) >= self.task.max_conversation_depth or any(
            prob >= self.task.confidence_threshold
            for prob in node.belief_state.values()
        )


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


@deprecated("Implement prior estimation per task instead")
def get_uniform_belief_state(hypothesis_space: list[str]) -> dict[str, float]:
    prob = 1.0 / len(hypothesis_space)
    return {item: prob for item in hypothesis_space}
