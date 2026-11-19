#!/usr/bin/env python3
"""Deterministic tests for the graph training contract."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

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
    print("[PASS] Graph training contract tests succeeded!")


if __name__ == "__main__":
    main()
