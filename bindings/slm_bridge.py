"""SLM 3B Verbalization Bridge for AE Arch Engine (`AE::mind1_v1.1_Stable`)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional

from ae_engine import AEArchEngine, AEInferenceOutput


class AESLMOrchestrator:
    def __init__(
        self,
        engine: Optional[AEArchEngine] = None,
        endpoint_url: Optional[str] = None,
        model_name: str = "qwen2.5:3b-instruct",
    ) -> None:
        self.engine = engine or AEArchEngine()
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    def answer(self, query: str, target_hint: int = -1, reroot_node: Optional[int] = None) -> Dict[str, Any]:
        if reroot_node is not None:
            self.engine.reroot(reroot_node)

        ae_out: AEInferenceOutput = self.engine.infer(query, target_hint=target_hint)

        if self.endpoint_url:
            slm_reply = self._call_remote_slm(ae_out.ast_prompt)
        else:
            slm_reply = self._deterministic_verbalize(ae_out)

        return {
            "query": query,
            "trace": ae_out.explainable_trace,
            "latency_us": ae_out.latency_us,
            "estimated_ast_tokens": ae_out.estimated_tokens,
            "optimal_path": ae_out.optimal_path,
            "time_ray_forward": ae_out.time_ray.forward_cascade_score,
            "time_ray_backward": ae_out.time_ray.backward_consistency_score,
            "ast_prompt": ae_out.ast_prompt,
            "verbalized_response": slm_reply,
        }

    def _call_remote_slm(self, ast_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model_name,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты детерминированный вербализатор AE Arch Engine. Отвечай строго по узлам THOUGHT_NODE.",
                },
                {"role": "user", "content": ast_prompt},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

    @staticmethod
    def _deterministic_verbalize(out: AEInferenceOutput) -> str:
        chunks = [f"KB_CHUNK_{out.centroid_id * 100 + u}" for u in out.optimal_path]
        return (
            f"[AE::mind1 Deterministic Verbalizer] Локализован центроид #{out.centroid_id} "
            f"(за {out.centroid_hops} шага O(log N)). По булевой маске {out.morpheme_mask_hex} "
            f"(найдено морфем: {out.matched_morphemes}) построен поток Диница (max_flow={out.max_flow}) "
            f"по траектории узлов {out.optimal_path} (источники: {', '.join(chunks)}). "
            f"Асинхронная проверка Time-Ray: Forward={out.time_ray.forward_cascade_score:.2f}, "
            f"Backward={out.time_ray.backward_consistency_score:.2f}."
        )
