"""Logic-task benchmarks for the AE core algorithms.

Every task compares a core algorithm against an independent brute-force
reference implemented here, so a regression in the C++ code fails the metric
instead of silently changing behavior.

ADR: docs/adr/0013-benchmarks-and-verification.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from training.ae_core import load_core


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


# ADR-0005: docs/adr/0005-thought-graph-mst-lca.md
def mst_weight_map(tg) -> dict[tuple[int, int], int]:
    weights: dict[tuple[int, int], int] = {}
    for node in range(tg.n):
        for neighbor, weight in tg.adj_mst[node]:
            key = (min(node, neighbor), max(node, neighbor))
            weights.setdefault(key, weight)
    return weights


# ADR-0005: docs/adr/0005-thought-graph-mst-lca.md
def bfs_path(adjacency: dict[int, list[int]], source: int, target: int) -> list[int]:
    parent = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if target not in parent:
        return []
    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    return list(reversed(path))


def adjacency_lists(tg) -> dict[int, list[int]]:
    return {node: [neighbor for neighbor, _ in tg.adj_mst[node]] for node in range(tg.n)}


# ADR-0005: docs/adr/0005-thought-graph-mst-lca.md
def kruskal_max_weight(n: int, edges: list[tuple[int, int, int]]) -> tuple[int, int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    used = 0
    for u, v, weight in sorted(edges, key=lambda e: -e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            total += weight
            used += 1
    return used, total


# ADR-0004: docs/adr/0004-dinitz-max-flow-decoder.md
# Независимый эталон для dinitz_matches_edmonds_karp.
def edmonds_karp(n: int, edges: list[tuple[int, int, int]], source: int, sink: int) -> int:
    capacity = [[0] * n for _ in range(n)]
    for u, v, cap in edges:
        capacity[u][v] += cap
    flow = 0
    while True:
        parent = [-1] * n
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for neighbor in range(n):
                if parent[neighbor] == -1 and capacity[node][neighbor] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if parent[sink] == -1:
            return flow
        bottleneck = 10 ** 9
        node = sink
        while node != source:
            bottleneck = min(bottleneck, capacity[parent[node]][node])
            node = parent[node]
        node = sink
        while node != source:
            capacity[parent[node]][node] -= bottleneck
            capacity[node][parent[node]] += bottleneck
            node = parent[node]
        flow += bottleneck


# ADR-0013: docs/adr/0013-benchmarks-and-verification.md
def run(rag_nodes: int = 20000, seed: int = 42, pairs: int = 200) -> dict[str, Any]:
    core = load_core()
    engine = core.AEEnginePipeline(rag_nodes, 6)
    tg = engine.thought_graph()
    trie = engine.trie()
    rag = engine.rag()
    decoder = engine.decoder()
    rng = random.Random(seed)
    results: list[TaskResult] = []

    weights = mst_weight_map(tg)
    adjacency = adjacency_lists(tg)
    node_pairs = [(rng.randrange(tg.n), rng.randrange(tg.n)) for _ in range(pairs)]

    # 1. path_between == BFS path in the MST
    task = TaskResult("path_matches_bfs")
    for u, v in node_pairs:
        expected = bfs_path(adjacency, u, v)
        actual = list(tg.path_between(u, v))
        ok = actual == expected
        if ok:
            for a, b in zip(actual, actual[1:]):
                ok = ok and (min(a, b), max(a, b)) in weights
        task.total += 1
        task.passed += int(ok)
    results.append(task)

    # 2. query_path_binup returns the true maximum edge weight on the path
    task = TaskResult("path_max_weight_matches_bruteforce")
    for u, v in node_pairs:
        path = bfs_path(adjacency, u, v)
        expected = max(weights[(min(a, b), max(a, b))] for a, b in zip(path, path[1:])) if len(path) > 1 else 0
        _, actual = tg.query_path_binup(u, v)
        task.total += 1
        task.passed += int(actual == expected)
    results.append(task)

    # 3. query_path_min_binup returns the true bottleneck
    task = TaskResult("path_min_weight_matches_bruteforce")
    for u, v in node_pairs:
        if u == v:
            continue  # путь из одной вершины не содержит рёбер
        path = bfs_path(adjacency, u, v)
        expected = min(weights[(min(a, b), max(a, b))] for a, b in zip(path, path[1:]))
        _, actual = tg.query_path_min_binup(u, v)
        task.total += 1
        task.passed += int(actual == expected)
    results.append(task)

    # 4. reroot keeps the undirected path and its weights invariant
    task = TaskResult("reroot_preserves_paths")
    baseline = {pair: (set(tg.path_between(*pair)), tg.query_path_min_binup(*pair)[1], tg.query_path_binup(*pair)[1])
                for pair in node_pairs[:60]}
    for new_root in (1, 7, tg.n // 2, tg.n - 1):
        tg.reroot(new_root)
        for pair, (nodes, min_w, max_w) in baseline.items():
            current = (set(tg.path_between(*pair)), tg.query_path_min_binup(*pair)[1], tg.query_path_binup(*pair)[1])
            task.total += 1
            task.passed += int(current == (nodes, min_w, max_w))
    tg.reroot(0)
    results.append(task)

    # 5. Kruskal: spanning tree shape and total weight
    edges = [(edge.u, edge.v, edge.weight) for edge in tg.all_edges]
    expected_edges, expected_weight = kruskal_max_weight(tg.n, edges)
    actual_weight = sum(weights.values())
    actual_edges = len(weights)
    task = TaskResult("kruskal_maximum_spanning_tree")
    task.total = 2
    task.passed = int(actual_edges == expected_edges == tg.n - 1) + int(actual_weight == expected_weight)
    task.details = {"edges": actual_edges, "weight": actual_weight, "reference_weight": expected_weight}
    results.append(task)

    # 6. Dinitz max-flow == Edmonds-Karp on random graphs
    task = TaskResult("dinitz_matches_edmonds_karp")
    for _ in range(40):
        n = 8
        graph_edges: list[tuple[int, int, int]] = []
        flow_decoder = core.DinitzGraphDecoder(n)
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.25:
                    cap = rng.randint(1, 12)
                    graph_edges.append((u, v, cap))
                    flow_decoder.add_static_edge(u, v, cap, 0)
        flow_decoder.apply_boolean_mask_and_reset(1)
        actual = flow_decoder.max_flow(0, n - 1)
        expected = edmonds_karp(n, graph_edges, 0, n - 1)
        task.total += 1
        task.passed += int(actual == expected)
    results.append(task)

    # 7. centroid decomposition bounds
    task = TaskResult("centroid_lookup_bounds")
    max_hops = 0
    for target in range(0, rag.n_nodes, max(1, rag.n_nodes // 200)):
        centroid, hops = rag.locate_centroid_cluster(target)
        max_hops = max(max_hops, hops)
        ok = 0 <= centroid < rag.n_nodes and hops <= 4
        if ok:
            depth = rag.centroid_depth[centroid] if centroid < len(rag.centroid_depth) else 0
            ok = depth <= math.ceil(math.log2(max(rag.n_nodes, 2))) + 1
        task.total += 1
        task.passed += int(ok)
    task.details = {"max_hops": max_hops}
    results.append(task)

    # 8. trie: known roots carry the expected semantic cluster
    expected_clusters = {
        "граф": "GRAPH", "знание": "KNOW", "поток": "FLOW", "текст": "TEXT",
        "число": "NUM", "нейрон": "MIND", "программа": "COMPUTE",
        "сеть": "NET", "сортировка": "ACTION",
        "graph": "GRAPH", "knowledge": "KNOW", "flow": "FLOW", "text": "TEXT",
        "number": "NUM", "neuron": "MIND", "program": "COMPUTE",
        "protocol": "NET", "search": "ACTION",
    }
    order = ["GRAPH", "KNOW", "FLOW", "TEXT", "NUM", "TIME", "SPACE", "MIND",
             "COMPUTE", "NET", "PHYS", "CHEM", "BIO", "MED", "ECON", "LAW",
             "SOC", "QUALITY", "NATURE", "ACTION"]
    neutral = [0.0] * 16
    task = TaskResult("trie_known_root_clusters")
    for word, cluster in expected_clusters.items():
        mask, _ = trie.compile_morpheme_mask(word, neutral)
        bit = 1 << (28 + order.index(cluster))
        task.total += 1
        task.passed += int(bool(mask & bit))
    results.append(task)

    # 9. unknown words must not invent semantic clusters
    task = TaskResult("trie_unknown_words_clean")
    for word in ("кот", "zzz", "qwerty", "абракадабра", "xylophonezz"):
        mask, count = trie.compile_morpheme_mask(word, neutral)
        cluster_bits = mask & 0x0000FFFFF0000000
        task.total += 1
        task.passed += int(cluster_bits == 0)
    results.append(task)

    # 10. flow / bottleneck invariants on real queries
    task = TaskResult("flow_bottleneck_invariants")
    for query in ("граф знаний", "поток данных", "нейрон и сознание", "", "unknown token"):
        result = engine.infer(query, -1, False)
        ok = result.path_bottleneck <= result.max_flow
        ok = ok and len(result.optimal_path) >= 2
        ok = ok and result.optimal_path[0] == 0 and result.optimal_path[-1] == tg.n - 1
        ok = ok and result.max_flow > 0
        task.total += 1
        task.passed += int(ok)
    results.append(task)

    return {
        "suite": "logic",
        "seed": seed,
        "rag_nodes": rag_nodes,
        "tasks": [task.as_dict() for task in results],
        "overall_pass_rate": round(sum(t.passed for t in results) / max(1, sum(t.total for t in results)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-nodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()
    report = run(args.rag_nodes, args.seed, args.pairs)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for task in report["tasks"]:
            print(f"{task['name']:42s} {task['passed']:4d}/{task['total']:<4d} {task['pass_rate']:.3f}")
        print(f"{'OVERALL':42s} {'':4s} {'':4s} {report['overall_pass_rate']:.3f}")


if __name__ == "__main__":
    main()
