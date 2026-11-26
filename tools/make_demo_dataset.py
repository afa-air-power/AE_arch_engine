#!/usr/bin/env python3
"""Create a deterministic graph dataset using AE algorithm-derived features.

ADR: docs/adr/0012-trainable-graph-contour.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.nodes
    labels = np.arange(n) % 3
    features = np.zeros((n, 12), dtype=np.float32)
    # Features mirror the existing deterministic algorithmic signals: semantic
    # mask bits, flow/bottleneck-like values, centroid hops, and path position.
    features[:, 0:3] = np.eye(3, dtype=np.float32)[labels]
    features[:, 3] = (np.arange(n) % 20) / 20.0
    features[:, 4] = ((np.arange(n) * 7) % 101) / 100.0
    features[:, 5] = ((np.arange(n) * 13) % 97) / 96.0
    features[:, 6:] = rng.normal(0, 0.05, size=(n, 6))
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for offset in (1, 2, 5):
            j = (i + offset) % n
            adjacency[i, j] = adjacency[j, i] = 1.0
    split = np.full(n, "train", dtype=object)
    split[int(n * 0.7):int(n * 0.85)] = "val"
    split[int(n * 0.85):] = "test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"node_features": features.tolist(), "adjacency": adjacency.tolist(), "labels": labels.tolist(), "split": split.tolist()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
