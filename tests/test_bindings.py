#!/usr/bin/env python3
"""Integration tests for Python ctypes SDK (`ae_engine.py`) and SLM Bridge (`slm_bridge.py`)."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bindings"))

from ae_engine import AEArchEngine
from slm_bridge import AESLMOrchestrator


def main() -> None:
    with AEArchEngine(rag_nodes=20000, rag_avg_degree=6) as engine:
        engine.add_morpheme("нейро", 22, AEArchEngine.MORPHEME_ROOT, 0x0001FFFFFFFFFFF)
        res = engine.infer(
            "переподключение графа мыслей к центроидному дереву памяти и генерация аст токенов",
            target_hint=14158,
        )
        assert res.matched_morphemes >= 10
        assert res.max_flow > 0
        assert len(res.optimal_path) >= 2
        assert "AE_VERIFIED_AST_V1.1" in res.ast_prompt
        assert res.nl_text, "natural-language output must not be empty"
        assert res.nl_consistent
        assert res.nl_cluster_recall >= 0.99
        assert res.nl_reparsed_morphemes > 0
        assert "Обратный разбор" in res.nl_feedback

        engine.reroot(8)
        res_reroot = engine.infer("нейро граф мыслей", target_hint=500)
        assert ":root 8" in res_reroot.ast_prompt

        orchestrator = AESLMOrchestrator(engine=engine)
        ans = orchestrator.answer("разреженный граф знаний и поток Диница", target_hint=1024, reroot_node=0)
        assert ans["nl_consistent"]
        assert ans["natural_language"]
        assert "KB_CHUNK_" in ans["trace_response"]

        print("[PASS] Python bindings & SLM bridge integration tests succeeded!")


if __name__ == "__main__":
    main()
