#!/usr/bin/env python3
"""Contract tests for the labeled text-cluster corpora."""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.merge_corpora import merge

CORPORA = ROOT / "training" / "corpora"
SEED = CORPORA / "text_clusters.json"
MANUAL = CORPORA / "text_clusters_manual.json"
EXPANDED = CORPORA / "text_clusters_expanded.json"


def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict, path: pathlib.Path) -> None:
    labels = payload["labels"]
    samples = payload["samples"]
    assert len(labels) == len(set(labels)), f"{path}: duplicate labels"
    assert samples, f"{path}: empty corpus"
    seen = set()
    for sample in samples:
        assert sample["query"].strip(), f"{path}: empty query"
        label = int(sample["label"])
        assert 0 <= label < len(labels), f"{path}: label out of range"
        assert sample["cluster"] == labels[label], (
            f"{path}: cluster {sample['cluster']} != labels[{label}]"
        )
        assert sample["split"] in {"train", "val", "test"}, f"{path}: bad split"
        key = sample["query"].strip().lower()
        assert key not in seen, f"{path}: duplicate query {sample['query']!r}"
        seen.add(key)
    # Every class must be represented in train and in at least one of val/test.
    for label in range(len(labels)):
        splits = Counter(s["split"] for s in samples if int(s["label"]) == label)
        assert splits["train"] > 0, f"{path}: class {label} has no train samples"
        assert splits["val"] + splits["test"] > 0, (
            f"{path}: class {label} has no held-out samples"
        )


def main() -> None:
    seed = load(SEED)
    manual = load(MANUAL)
    validate(seed, SEED)
    validate(manual, MANUAL)

    assert len(seed["samples"]) == 90
    assert len(manual["samples"]) == 180
    per_cluster = Counter(s["cluster"] for s in manual["samples"])
    assert set(per_cluster.values()) == {20}, per_cluster
    for cluster in manual["labels"]:
        splits = Counter(
            s["split"] for s in manual["samples"] if s["cluster"] == cluster
        )
        assert splits == {"train": 12, "val": 4, "test": 4}, (cluster, splits)

    merged = merge([SEED, MANUAL], "text_clusters_expanded")
    validate(merged, pathlib.Path("<merged>"))
    assert len(merged["samples"]) == 270
    assert merged["meta"]["duplicates_skipped"] == 0

    if EXPANDED.exists():
        on_disk = load(EXPANDED)
        validate(on_disk, EXPANDED)
        assert on_disk["samples"] == merged["samples"], (
            "expanded corpus is stale: re-run tools/merge_corpora.py"
        )

    print("[PASS] Labeled corpus contract tests succeeded!")


if __name__ == "__main__":
    main()
