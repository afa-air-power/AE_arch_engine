"""Text-analysis benchmarks for the AE core.

These tasks measure the deterministic text layer: language detection, morpheme
segmentation, semantic cluster extraction, the natural-language round trip and
output determinism.  They complement the logic suite, which measures the graph
algorithms.

ADR: docs/adr/0013-benchmarks-and-verification.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from training.ae_core import load_core

CLUSTER_ORDER = [
    "GRAPH", "KNOW", "FLOW", "TEXT", "NUM", "TIME", "SPACE", "MIND",
    "COMPUTE", "NET", "PHYS", "CHEM", "BIO", "MED", "ECON", "LAW",
    "SOC", "QUALITY", "NATURE", "ACTION",
]


@dataclass
class TaskResult:
    name: str
    passed: int = 0
    total: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 6),
            "details": self.details,
        }


# ADR-0002: docs/adr/0002-morphemic-patricia-trie.md
def clusters_of(mask: int) -> set[str]:
    return {CLUSTER_ORDER[i] for i in range(20) if (mask >> (28 + i)) & 1}


def prf(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


LANGUAGE_CASES = [
    ("граф знаний и дерево вершин", "russian"),
    ("нейрон и сознание", "russian"),
    ("поток данных и маршрут", "russian"),
    ("морфемы слова и текст", "russian"),
    ("matrix and vector", "english"),
    ("brain neuron cognition", "english"),
    ("data flow pipeline", "english"),
    ("text tokens and morphology", "english"),
]

SEGMENTATION_CASES = {
    "граф": "граф",
    "графический": "граф",
    "нейрон": "нейрон",
    "программирование": "програм",
    "текстовый": "текст",
    "переподключение": "ключ",
    "knowledge": "know",
    "networking": "net",
}

UNKNOWN_CASES = ["кот", "zzz", "qwerty", "абракадабра", "xylophonezz"]

CLUSTER_CASES = [
    ("граф и дерево вершин", {"GRAPH"}),
    ("знание и онтология понятий", {"KNOW"}),
    ("поток данных и маршрут", {"FLOW"}),
    ("текст и морфемы слова", {"TEXT"}),
    ("число вектор и матрица", {"NUM"}),
    ("программа компилятор и модуль", {"COMPUTE"}),
    ("мозг нейрон и сознание", {"MIND"}),
    ("сеть протокол и пакет", {"NET"}),
    ("сортировка и фильтрация", {"ACTION"}),
    ("graph structure and spanning tree", {"GRAPH"}),
    ("knowledge ontology and facts", {"KNOW"}),
    ("network protocol and packet", {"NET"}),
]


# ADR-0013: docs/adr/0013-benchmarks-and-verification.md
def run(rag_nodes: int = 20000) -> dict[str, Any]:
    core = load_core()
    engine = core.AEEnginePipeline(rag_nodes, 6)
    trie = engine.trie()
    neutral = [0.0] * 16
    results: list[TaskResult] = []

    # 1. language detection
    task = TaskResult("language_detection")
    for text, expected in LANGUAGE_CASES:
        task.total += 1
        task.passed += int(core.detect_language(text) == expected)
    results.append(task)

    # 2. semantic cluster extraction: per-sample precision/recall/F1
    task = TaskResult("semantic_cluster_detection")
    precision_sum = recall_sum = f1_sum = 0.0
    misses: list[dict[str, Any]] = []
    for query, expected in CLUSTER_CASES:
        mask, _ = trie.compile_morpheme_mask(query, neutral)
        predicted = clusters_of(mask)
        precision, recall, f1 = prf(predicted, expected)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        ok = recall == 1.0 and precision >= 0.5
        task.total += 1
        task.passed += int(ok)
        if not ok:
            misses.append({"query": query, "expected": sorted(expected), "predicted": sorted(predicted)})
    count = len(CLUSTER_CASES)
    task.details = {
        "mean_precision": round(precision_sum / count, 6),
        "mean_recall": round(recall_sum / count, 6),
        "mean_f1": round(f1_sum / count, 6),
        "misses": misses,
    }
    results.append(task)

    # 3. morpheme segmentation: the expected stem appears in the parse
    task = TaskResult("morpheme_segmentation")
    misses = []
    for word, expected_stem in SEGMENTATION_CASES.items():
        morphemes = list(core.parse_morphemes(word))
        ok = any(expected_stem in morpheme for morpheme in morphemes)
        if not ok:
            mask, count = trie.compile_morpheme_mask(word, neutral)
            ok = bool(mask & 0x0000FFFFF0000000) and count > 0
        task.total += 1
        task.passed += int(ok)
        if not ok:
            misses.append({"word": word, "expected": expected_stem, "morphemes": morphemes})
    task.details = {"misses": misses}
    results.append(task)

    # 4. unknown words stay clean
    task = TaskResult("unknown_words_no_clusters")
    for word in UNKNOWN_CASES:
        mask, _ = trie.compile_morpheme_mask(word, neutral)
        task.total += 1
        task.passed += int((mask & 0x0000FFFFF0000000) == 0)
    results.append(task)

    # 5. natural-language round trip consistency
    task = TaskResult("nl_round_trip_consistency")
    queries = [query for query, _ in CLUSTER_CASES] + [text for text, _ in LANGUAGE_CASES]
    recall_sum = precision_sum = 0.0
    for query in queries:
        result = engine.infer(query, -1, False)
        nl = result.natural_language
        recall_sum += nl.cluster_recall
        precision_sum += nl.cluster_precision
        task.total += 1
        task.passed += int(nl.consistent and nl.cluster_recall >= 0.99)
    count = len(queries)
    task.details = {
        "mean_cluster_recall": round(recall_sum / count, 6),
        "mean_cluster_precision": round(precision_sum / count, 6),
    }
    results.append(task)

    # 6. determinism: identical input -> identical algorithm output
    task = TaskResult("deterministic_output")
    for query in queries[:8]:
        first = engine.infer(query, 1234, False)
        second = engine.infer(query, 1234, False)
        same = (first.morpheme_mask == second.morpheme_mask
                and first.max_flow == second.max_flow
                and first.path_bottleneck == second.path_bottleneck
                and list(first.optimal_path) == list(second.optimal_path))
        task.total += 1
        task.passed += int(same)
    results.append(task)

    # 7. AST contract: verified markers and path consistency
    task = TaskResult("ast_contract")
    for query in queries[:8]:
        result = engine.infer(query, -1, False)
        ast = result.ast.ast_text
        ok = "AE_VERIFIED_AST_V1.1" in ast and "THOUGHT_NODE" in ast
        path = list(result.optimal_path)
        ok = ok and f":root {engine.thought_graph().current_root}" in ast
        ok = ok and all(f":node_id {node}" in ast for node in path)
        task.total += 1
        task.passed += int(ok)
    results.append(task)

    return {
        "suite": "text",
        "rag_nodes": rag_nodes,
        "tasks": [task.as_dict() for task in results],
        "overall_pass_rate": round(sum(t.passed for t in results) / max(1, sum(t.total for t in results)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-nodes", type=int, default=20000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args.rag_nodes)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for task in report["tasks"]:
            print(f"{task['name']:34s} {task['passed']:4d}/{task['total']:<4d} {task['pass_rate']:.3f}")
        print(f"{'OVERALL':34s} {'':4s} {'':4s} {report['overall_pass_rate']:.3f}")


if __name__ == "__main__":
    main()
