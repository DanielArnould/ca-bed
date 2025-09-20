import asyncio
from dataclasses import dataclass, field
import logging

from sentence_transformers import SentenceTransformer
from voyager import Index, Space

LOGGER = logging.getLogger(__name__)


@dataclass
class Cluster:
    questions: dict[str, int]
    # Answer -> hypothesis -> likelihood
    likelihoods: dict[str, dict[str, float]] | None
    lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())


class QuestionClustering:
    index: Index
    clusters: dict[int, Cluster]
    threshold: float
    model: SentenceTransformer

    def __init__(self, threshold: float):
        LOGGER.info(f"Setting up question cluster with threshold '{threshold}'")
        self.model = SentenceTransformer(
            "quora-distilbert-multilingual",
            backend="onnx",
            model_kwargs={"file_name": "onnx/model_qint8_avx512.onnx"},
        )
        self.index = Index(Space.Cosine, num_dimensions=768)
        self.threshold = threshold
        self.clusters = {}

    def get_cluster(self, question: str) -> Cluster:
        embedding = self.model.encode(
            question, convert_to_numpy=True, normalize_embeddings=False
        )
        neighbours, distances = (
            self.index.query(embedding, k=1) if len(self.clusters) >= 1 else ([], [])
        )

        if len(neighbours) > 0 and 1 - distances[0] >= self.threshold:
            best_cluster = self.clusters[neighbours[0]]
            LOGGER.info(
                f"Cluster found for {question}, with similarity {1 - distances[0]}!"
            )
            best_cluster.questions[question] = (
                best_cluster.questions.get(question, 0) + 1
            )
            return best_cluster

        LOGGER.info(f"Cluster not found for {question}. Creating new cluster...")
        idx = self.index.add_item(embedding)
        new_cluster = Cluster(
            {question: 1},
            None,
        )
        self.clusters[idx] = new_cluster
        return new_cluster
