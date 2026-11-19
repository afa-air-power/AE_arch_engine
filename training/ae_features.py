"""Algorithm-derived feature extraction and dataset assembly.

Node features are produced by the C++ core itself: Taylor-Intent Kernel,
morphemic Patricia trie mask, Mega RAG centroid decomposition, Dinitz
max-flow, MST/LCA thought-graph path statistics, Time-Ray smoothing and the
natural-language round trip.  The Python layer only batches them into a
trainable dataset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from training.ae_core import load_core

FEATURE_DIM = 76

# Группы признаков. intent/centroid/timeray выводятся из байтового хэша запроса
# и кодируют его идентичность, а не смысл, поэтому по умолчанию исключены:
# на малой выборке они дают запоминание вместо обобщения.
FEATURE_GROUPS: dict[str, range] = {
    "intent": range(0, 16),
    "morphology": range(16, 25),
    "centroid": range(25, 28),
    "flow": range(28, 32),
    "topology": range(32, 42),
    "timeray": range(42, 46),
    "nl": range(46, 51),
    "query": range(51, 56),
    "clusters": range(56, 76),
}

DEFAULT_GROUPS: tuple[str, ...] = (
    "morphology", "flow", "topology", "nl", "query", "clusters",
)


def group_columns(groups: Iterable[str]) -> list[int]:
    columns: list[int] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {group}")
        columns.extend(FEATURE_GROUPS[group])
    return sorted(set(columns))


@dataclass
class AlgorithmFeatureExtractor:
    """Thin wrapper over ``AEEnginePipeline::extract_features``."""

    rag_nodes: int = 20000
    rag_avg_degree: int = 6
    core: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.core is None:
            self.core = load_core()
        self.engine = self.core.AEEnginePipeline(self.rag_nodes, self.rag_avg_degree)

    @property
    def feature_names(self) -> list[str]:
        return list(self.core.AlgorithmFeatureVector.names())

    def extract(self, query: str, target_hint: int = -1) -> np.ndarray:
        vector = self.engine.extract_features(query, int(target_hint))
        return np.asarray(vector.values, dtype=np.float32)

    def infer(self, query: str, target_hint: int = -1):
        return self.engine.infer(query, int(target_hint), False)

    def reroot(self, node: int) -> None:
        self.engine.reroot_thought_graph(int(node))

    @property
    def thought_graph(self):
        return self.engine.thought_graph()

    @property
    def trie(self):
        return self.engine.trie()

    @property
    def rag(self):
        return self.engine.rag()

    @property
    def decoder(self):
        return self.engine.decoder()


def load_corpus(path: str | pathlib.Path) -> list[dict[str, Any]]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    if not samples:
        raise ValueError("corpus must contain at least one sample")
    for sample in samples:
        if "query" not in sample or "label" not in sample:
            raise ValueError("each sample needs 'query' and 'label'")
        sample.setdefault("split", "train")
        if sample["split"] not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {sample['split']}")
    return samples


def _tokens(query: str) -> set[str]:
    out: set[str] = set()
    current: list[str] = []
    for ch in query.lower():
        if ch.isalnum() or ch == "_":
            current.append(ch)
        elif current:
            out.add("".join(current))
            current = []
    if current:
        out.add("".join(current))
    return out


def lexical_adjacency(queries: list[str], min_similarity: float = 0.15) -> np.ndarray:
    """Content-derived graph: edges between queries with shared vocabulary.

    Unlike kNN over the feature vector this does not use labels, and it groups
    paraphrases of the same topic, which is exactly what graph smoothing should
    exploit.
    """
    n = len(queries)
    tokens = [_tokens(query) for query in queries]
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            union = tokens[i] | tokens[j]
            if not union:
                continue
            similarity = len(tokens[i] & tokens[j]) / len(union)
            if similarity >= min_similarity:
                adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def knn_adjacency(features: np.ndarray, neighbors: int = 4) -> np.ndarray:
    """Symmetric kNN graph over standardized algorithm feature vectors."""
    n = features.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if neighbors <= 0:
        return np.zeros((n, n), dtype=np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = (features - mean) / std
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    normalized = normalized / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k = max(1, min(int(neighbors), n - 1))
    adjacency = np.zeros((n, n), dtype=np.float32)
    for row in range(n):
        order = np.argsort(-similarity[row], kind="stable")[:k]
        adjacency[row, order] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def build_adjacency(
    features: np.ndarray,
    queries: list[str],
    graph: str = "lexical",
    neighbors: int = 0,
    min_similarity: float = 0.15,
) -> np.ndarray:
    if graph == "none":
        return np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)
    if graph == "lexical":
        return lexical_adjacency(queries, min_similarity)
    if graph == "knn":
        return knn_adjacency(features, neighbors)
    raise ValueError(f"unknown graph mode: {graph}")


def build_dataset(
    samples: Iterable[dict[str, Any]],
    extractor: AlgorithmFeatureExtractor | None = None,
    neighbors: int = 0,
    rag_nodes: int = 20000,
    groups: Iterable[str] = DEFAULT_GROUPS,
    graph: str = "none",
    min_similarity: float = 0.15,
) -> dict[str, Any]:
    samples = list(samples)
    if extractor is None:
        extractor = AlgorithmFeatureExtractor(rag_nodes=rag_nodes)
    columns = group_columns(groups)
    all_names = extractor.feature_names
    features = np.stack([extractor.extract(sample["query"], sample.get("target_hint", -1)) for sample in samples])
    features = features[:, columns]
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    split = np.asarray([sample["split"] for sample in samples], dtype=object)
    queries = [sample["query"] for sample in samples]
    adjacency = build_adjacency(features, queries, graph=graph, neighbors=neighbors,
                                min_similarity=min_similarity)
    return {
        "node_features": features.tolist(),
        "adjacency": adjacency.tolist(),
        "labels": labels.tolist(),
        "split": split.tolist(),
        "queries": queries,
        "feature_names": [all_names[i] for i in columns],
        "meta": {
            "feature_dim": int(features.shape[1]),
            "feature_groups": list(groups),
            "graph": graph,
            "neighbors": int(neighbors),
            "min_similarity": float(min_similarity),
            "rag_nodes": int(rag_nodes),
            "source": "ae_core_algorithm_features",
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="JSON corpus with samples[{query,label,split}]")
    parser.add_argument("output", help="destination dataset JSON")
    parser.add_argument("--neighbors", type=int, default=0)
    parser.add_argument("--graph", default="none", choices=("lexical", "knn", "none"),
                        help="graph structure over samples; 'none' is the measured best on the seed corpus")
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--rag-nodes", type=int, default=20000)
    parser.add_argument("--target-hint", type=int, default=-1)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS),
                        help=f"comma-separated feature groups: {', '.join(FEATURE_GROUPS)}")
    args = parser.parse_args(argv)

    samples = load_corpus(args.corpus)
    groups = tuple(part.strip() for part in args.groups.split(",") if part.strip())
    extractor = AlgorithmFeatureExtractor(rag_nodes=args.rag_nodes)
    dataset = build_dataset(samples, extractor=extractor, neighbors=args.neighbors,
                            rag_nodes=args.rag_nodes, groups=groups, graph=args.graph,
                            min_similarity=args.min_similarity)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(json.dumps({
        "samples": len(samples),
        "feature_dim": dataset["meta"]["feature_dim"],
        "feature_groups": dataset["meta"]["feature_groups"],
        "split": {name: int(np.sum(np.asarray(dataset["split"]) == name)) for name in ("train", "val", "test")},
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
