#!/usr/bin/env python3
"""Merge manually labeled corpora into one training corpus.

The seed corpus produced by ``tools/make_text_cluster_corpus.py`` is generated
and must not be edited by hand.  Manual labels live in separate files (for
example ``training/corpora/text_clusters_manual.json``) and are merged here.

Contract:
- every input uses the same ``labels`` list;
- every sample has ``query`` and ``label``; ``cluster`` is derived if missing;
- normalized duplicate queries are collapsed (same label) or rejected
  (different labels);
- the merged file keeps the same schema consumed by ``training/ae_features.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Iterable


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _load(path: pathlib.Path) -> tuple[list[str], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        labels = list(payload.get("labels", []))
        samples = list(payload.get("samples", []))
    elif isinstance(payload, list):
        labels = []
        samples = list(payload)
    else:
        raise ValueError(f"{path}: expected an object or a list")
    if not samples:
        raise ValueError(f"{path}: no samples")
    return labels, samples


def merge(inputs: Iterable[pathlib.Path], name: str) -> dict[str, Any]:
    labels: list[str] | None = None
    merged: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    duplicates = 0
    sources: list[str] = []

    for path in inputs:
        file_labels, samples = _load(path)
        if labels is None:
            labels = file_labels
        elif file_labels and file_labels != labels:
            raise ValueError(
                f"{path}: labels {file_labels} do not match {labels}"
            )

        for sample in samples:
            query = str(sample["query"]).strip()
            label = int(sample["label"])
            key = _normalize_query(query)
            if not key:
                raise ValueError(f"{path}: empty query")
            if labels is not None and not (0 <= label < len(labels)):
                raise ValueError(f"{path}: label {label} out of range for {labels}")

            previous = seen.get(key)
            if previous is not None:
                if int(previous["label"]) != label:
                    raise ValueError(
                        f"{path}: duplicate query {query!r} has conflicting "
                        f"labels {previous['label']} and {label}"
                    )
                duplicates += 1
                continue

            cluster = sample.get("cluster")
            if not cluster and labels is not None:
                cluster = labels[label]
            split = sample.get("split", "train")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{path}: invalid split {split!r}")

            record = {
                "query": query,
                "label": label,
                "cluster": cluster,
                "split": split,
            }
            for extra in ("source", "notes", "target_hint"):
                if extra in sample:
                    record[extra] = sample[extra]
            merged.append(record)
            seen[key] = record

        sources.append(str(path))

    if labels is None:
        raise ValueError("no labels found in inputs")

    counts: dict[str, dict[str, int]] = {}
    for sample in merged:
        bucket = counts.setdefault(sample["cluster"], {"train": 0, "val": 0, "test": 0})
        bucket[sample["split"]] += 1

    return {
        "name": name,
        "labels": labels,
        "samples": merged,
        "meta": {
            "sources": sources,
            "samples": len(merged),
            "duplicates_skipped": duplicates,
            "per_cluster": counts,
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.add_argument("--name", default="text_clusters_expanded")
    args = parser.parse_args(argv)

    payload = merge(args.inputs, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta = payload["meta"]
    print(
        f"wrote {meta['samples']} samples to {args.output} "
        f"({meta['duplicates_skipped']} duplicates skipped)"
    )
    for cluster, split_counts in meta["per_cluster"].items():
        print(f"  {cluster:8s} {split_counts}")


if __name__ == "__main__":
    main()
