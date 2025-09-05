from dataclasses import dataclass
from operator import itemgetter
from sentence_transformers import SentenceTransformer, util
import torch


@dataclass
class Cluster:
    centroid: torch.Tensor
    likelihoods: dict[str, dict[str, float]]  # Answer -> hypothesis -> likelihood
    questions: list[str]

    def add_question(self, question: str, embedding: torch.Tensor) -> None:
        self.questions.append(question)
        self.centroid = torch.mean(torch.stack([self.centroid, embedding]), dim=0)


class QuestionClustering:
    clusters: list[Cluster]
    threshold: float
    model: SentenceTransformer

    def __init__(self, model_name="quora-distilbert-multilingual", threshold=0.95):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.clusters = []

    def get_embedding(self, question: str) -> torch.Tensor:
        return self.model.encode(question, convert_to_tensor=True)

    def get_nearest_cluster(self, embedding: torch.Tensor) -> Cluster | None:
        best_cluster, best_score = max(
            (
                (cluster, util.cos_sim(embedding, cluster.centroid).item())
                for cluster in self.clusters
            ),
            key=itemgetter(1),
            default=(None, -1),
        )
        return (
            best_cluster
            if best_cluster is not None and best_score >= self.threshold
            else None
        )

    def add_cluster(
        self,
        question: str,
        embedding: torch.Tensor,
        likelihoods: dict[str, dict[str, float]],
    ) -> None:
        self.clusters.append(Cluster(embedding, likelihoods, [question]))
