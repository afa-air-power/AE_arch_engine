"""Run every benchmark suite and emit one JSON report."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from benchmarks import logic_tasks, text_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-nodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "logic": logic_tasks.run(args.rag_nodes, args.seed, args.pairs),
        "text": text_tasks.run(args.rag_nodes),
    }
    overall = (
        sum(task["passed"] for suite in report.values() for task in suite["tasks"])
        / max(1, sum(task["total"] for suite in report.values() for task in suite["tasks"]))
    )
    report["overall_pass_rate"] = round(overall, 6)

    for name, suite in (("logic", report["logic"]), ("text", report["text"])):
        print(f"[{name}]")
        for task in suite["tasks"]:
            print(f"  {task['name']:40s} {task['passed']:4d}/{task['total']:<4d} {task['pass_rate']:.3f}")
    print(f"OVERALL pass rate: {report['overall_pass_rate']:.3f}")

    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {path}")
    return 0 if report["overall_pass_rate"] >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
