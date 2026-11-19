#!/usr/bin/env python3
"""Regression thresholds for the logic and text benchmark suites."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import logic_tasks, text_tasks


def main() -> None:
    logic = logic_tasks.run(rag_nodes=5000, seed=7, pairs=80)
    text = text_tasks.run(rag_nodes=5000)

    for suite in (logic, text):
        for task in suite["tasks"]:
            assert task["pass_rate"] >= 0.95, (suite["suite"], task)
        assert suite["overall_pass_rate"] >= 0.99, suite

    required = {
        "path_matches_bfs", "path_max_weight_matches_bruteforce",
        "path_min_weight_matches_bruteforce", "reroot_preserves_paths",
        "kruskal_maximum_spanning_tree", "dinitz_matches_edmonds_karp",
        "centroid_lookup_bounds", "trie_known_root_clusters",
        "trie_unknown_words_clean", "flow_bottleneck_invariants",
    }
    assert required <= {task["name"] for task in logic["tasks"]}

    required_text = {
        "language_detection", "semantic_cluster_detection", "morpheme_segmentation",
        "unknown_words_no_clusters", "nl_round_trip_consistency",
        "deterministic_output", "ast_contract",
    }
    assert required_text <= {task["name"] for task in text["tasks"]}

    print("[PASS] Logic and text benchmark suites meet thresholds!")


if __name__ == "__main__":
    main()
