#!/usr/bin/env python3
"""Deterministic tests for the graph training contract."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.ae_graph_train import GraphDataset, evaluate, load_checkpoint, train
from tools.make_demo_dataset import main as make_demo_dataset


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = pathlib.Path(tmp) / "dataset.json"
        checkpoint_path = pathlib.Path(tmp) / "checkpoint.json"
        make_demo_dataset_args = [str(dataset_path)]
        old_argv = sys.argv
        try:
            sys.argv = ["make_demo_dataset", *make_demo_dataset_args]
            make_demo_dataset()
        finally:
            sys.argv = old_argv
        dataset = GraphDataset.load(dataset_path)
        model, history = train(dataset, epochs=4, hidden_dim=8, checkpoint=checkpoint_path)
        assert len(history) == 4
        metrics = evaluate(model, dataset)
        assert set(metrics) == {"train", "val", "test"}
        restored = load_checkpoint(checkpoint_path)
        assert evaluate(restored, dataset) == metrics
        json.loads(checkpoint_path.read_text(encoding="utf-8"))

        # Empty adjacency means identity aggregation: training must work
        # without a dense n×n matrix for both linear and hidden models.
        n = 24
        features = np.zeros((n, 4), dtype=np.float32)
        labels = np.arange(n) % 3
        features[np.arange(n), labels] = 1.0
        split = np.array(["train"] * (n - 6) + ["val"] * 3 + ["test"] * 3, dtype=str)
        sparse = GraphDataset(features, np.zeros((0, 0), dtype=np.float32), labels, split)
        assert sparse.normalized_adjacency() is None
        for hidden in (0, 8):
            model, history = train(sparse, epochs=4, hidden_dim=hidden)
            assert len(history) == 4
            assert set(evaluate(model, sparse)) == {"train", "val", "test"}

    print("[PASS] Graph training contract tests succeeded!")


if __name__ == "__main__":
    main()
