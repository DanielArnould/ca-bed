from dataclasses import dataclass
import logging
from operator import itemgetter
from sentence_transformers import SentenceTransformer, util
import torch

LOGGER = logging.getLogger("Question Clustering")


@dataclass
class Cluster:
    centroid: torch.Tensor
    questions: list[str]
    # Answer -> hypothesis -> likelihood
    likelihoods: dict[str, dict[str, float]] | None

    def add_question(self, question: str) -> None:
        self.questions.append(question)


class QuestionClustering:
    clusters: list[Cluster]
    threshold: float
    model: SentenceTransformer
    model_name: str

    def __init__(
        self, threshold: float, model_name: str = "quora-distilbert-multilingual"
    ):
        LOGGER.info(
            f"Setting up question cluster with model '{model_name}' and threshold '{threshold}'"
        )
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.threshold = threshold
        self.clusters = []

    def get_cluster(self, question: str) -> Cluster:
        embedding = self.model.encode(question, convert_to_tensor=True)
        best_cluster, best_score = max(
            (
                (cluster, util.cos_sim(embedding, cluster.centroid).item())
                for cluster in self.clusters
            ),
            key=itemgetter(1),
            default=(None, -1),
        )

        if best_cluster is not None and best_score >= self.threshold:
            LOGGER.info(
                f"Cluster found for {question}. Adding to existing cluster of size -1"
            )
            best_cluster.add_question(question)
            return best_cluster

        LOGGER.info(f"Cluster not found for {question}. Creating new cluster...")
        new_cluster = Cluster(
            embedding,
            [question],
            None,
        )
        self.clusters.append(new_cluster)
        return new_cluster
