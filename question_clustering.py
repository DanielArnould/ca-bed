import asyncio
from dataclasses import dataclass, field
import logging

LOGGER = logging.getLogger("Question Clustering")


@dataclass
class Cluster:
    size: int
    # Answer -> hypothesis -> likelihood
    likelihoods: dict[str, dict[str, float]] | None
    lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())


class QuestionClustering:
    clusters: dict[str, Cluster]

    def __init__(self):
        LOGGER.info("Setting up question cluster...")
        self.clusters = {}

    def get_cluster(self, question: str) -> Cluster:
        cluster = self.clusters.get(question)

        if cluster is not None:
            LOGGER.info(
                f"Cluster found for {question}. Adding to existing cluster of size {cluster.size}"
            )
            cluster.size += 1
            return cluster

        LOGGER.info(f"Cluster not found for {question}. Creating new cluster...")
        new_cluster = Cluster(1, None)
        self.clusters[question] = new_cluster
        return new_cluster
