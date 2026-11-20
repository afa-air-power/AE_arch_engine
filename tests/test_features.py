#!/usr/bin/env python3
"""Contract tests for algorithm-derived features."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.ae_features import (
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    AlgorithmFeatureExtractor,
    build_dataset,
    group_columns,
    lexical_adjacency,
)


def main() -> None:
    extractor = AlgorithmFeatureExtractor(rag_nodes=2000)
    names = extractor.feature_names
    assert len(names) == 96, names
    assert names[0] == "intent_0"
    assert names[16] == "mask_popcount"
    assert names[28] == "max_flow"
    assert names[56] == "cluster_graph"
    assert names[75] == "cluster_action"
    assert names[76] == "cluster_count_graph"
    assert names[95] == "cluster_count_action"

    first = extractor.extract("граф знаний и поток данных")
    second = extractor.extract("граф знаний и поток данных")
    assert first.shape == (96,)
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0.0) and np.all(first <= 1.0)
    assert np.array_equal(first, second), "features must be deterministic"

    graph = extractor.extract("граф и дерево вершин")
    network = extractor.extract("сеть протокол и пакет")
    assert graph[56] == 1.0, "GRAPH cluster indicator must be set"
    assert network[65] == 1.0, "NET cluster indicator must be set"
    assert graph[65] == 0.0
    assert graph[76] >= network[76], "graph count must dominate a graph query"
    assert network[85] > 0.0, "NET cluster support must be counted"

    # Same cluster bits, different support: counts must separate the queries.
    coloring = extractor.extract("graph coloring and chromatic number")
    spectrum = extractor.extract("graph spectrum and matrix eigenvalue")
    assert np.array_equal(coloring[56:76], spectrum[56:76]), (
        "the example queries must share the same cluster mask"
    )
    assert not np.allclose(coloring[76:96], spectrum[76:96]), (
        "cluster counts must distinguish queries with an identical mask"
    )

    columns = group_columns(DEFAULT_GROUPS)
    assert 0 not in columns and 25 not in columns, "hash-derived groups excluded by default"
    assert 56 in columns and 76 in columns, "default model uses cluster bits and counts"
    assert len(columns) == 18
    assert all(group in FEATURE_GROUPS for group in DEFAULT_GROUPS)

    samples = [
        {"query": "граф и дерево", "label": 0, "split": "train"},
        {"query": "сеть и протокол", "label": 1, "split": "train"},
        {"query": "граф и вершины", "label": 0, "split": "test"},
    ]
    dataset = build_dataset(samples, extractor=extractor, graph="lexical")
    assert len(dataset["node_features"]) == 3
    assert len(dataset["node_features"][0]) == 18
    assert dataset["meta"]["feature_dim"] == 18
    adjacency = np.asarray(dataset["adjacency"])
    assert adjacency.shape == (3, 3)
    assert np.allclose(adjacency, adjacency.T)

    empty = build_dataset(samples, extractor=extractor, graph="none")
    assert np.asarray(empty["adjacency"]).size == 0, (
        "graph='none' must not materialize a dense n×n adjacency"
    )

    lexical = lexical_adjacency(["граф дерево", "граф вершина", "сеть протокол"], 0.2)
    assert lexical[0, 1] == 1.0
    assert lexical[0, 2] == 0.0

    print("[PASS] Algorithm feature contract tests succeeded!")


if __name__ == "__main__":
    main()
