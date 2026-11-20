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
MANUALS = sorted(CORPORA.glob("text_clusters_manual*.json"))
SYNTHETIC = CORPORA / "text_clusters_synthetic.json"
EXPANDED = CORPORA / "text_clusters_expanded.json"


def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict, path: pathlib.Path, require_holdout: bool = True) -> None:
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
        if require_holdout:
            assert splits["val"] + splits["test"] > 0, (
                f"{path}: class {label} has no held-out samples"
            )


def main() -> None:
    seed = load(SEED)
    validate(seed, SEED)
    assert len(seed["samples"]) == 90

    labels = seed["labels"]
    assert MANUALS, "no manual labeling waves found"
    for path in MANUALS:
        manual = load(path)
        validate(manual, path)
        assert manual["labels"] == labels, f"{path}: label order differs from seed"
        assert len(manual["samples"]) == 180, f"{path}: expected one wave of 180"
        per_cluster = Counter(s["cluster"] for s in manual["samples"])
        assert set(per_cluster.values()) == {20}, (path, per_cluster)
        for cluster in manual["labels"]:
            splits = Counter(
                s["split"] for s in manual["samples"] if s["cluster"] == cluster
            )
            assert splits == {"train": 12, "val": 4, "test": 4}, (path, cluster, splits)

    v2 = CORPORA / "text_clusters_manual_v2.json"
    if v2.exists():
        hard_notes = " ".join(
            s.get("notes", "") for s in load(v2)["samples"]
        )
        for pair in ("GRAPH/NET", "COMPUTE/ACTION", "KNOW/TEXT"):
            assert pair in hard_notes, f"manual-v2: no hard pair {pair}"

    merged_inputs = [SEED, *MANUALS]
    if SYNTHETIC.exists():
        synthetic = load(SYNTHETIC)
        validate(synthetic, SYNTHETIC, require_holdout=False)
        assert synthetic["labels"] == labels, "synthetic: label order differs from seed"
        assert len(synthetic["samples"]) == 4500, len(synthetic["samples"])
        per_cluster = Counter(s["cluster"] for s in synthetic["samples"])
        assert set(per_cluster.values()) == {500}, per_cluster
        assert all(s["split"] == "train" for s in synthetic["samples"]), (
            "synthetic data must stay in train so manual val/test remain clean"
        )
        assert all(s.get("source") == "synthetic-v1" for s in synthetic["samples"])
        for cluster in labels:
            rows = [s for s in synthetic["samples"] if s["cluster"] == cluster]
            assert Counter(s["language"] for s in rows) == {"ru": 250, "en": 250}, cluster
        synthetic_notes = " ".join(s.get("notes", "") for s in synthetic["samples"])
        for pair in ("GRAPH/NET", "COMPUTE/ACTION", "KNOW/TEXT",
                     "FLOW/NET", "MIND/NET", "NUM/GRAPH"):
            assert pair in synthetic_notes, f"synthetic: no hard pair {pair}"
        merged_inputs.append(SYNTHETIC)

    merged = merge(merged_inputs, "text_clusters_expanded")
    validate(merged, pathlib.Path("<merged>"))
    expected = 90 + 180 * len(MANUALS)
    if SYNTHETIC.exists():
        expected += len(load(SYNTHETIC)["samples"])
    assert len(merged["samples"]) == expected
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
