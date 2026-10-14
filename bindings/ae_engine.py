"""AE Arch Engine (AE::mind1_v1.1_Stable) — Python SDK & Zero-Dependency Ctypes Wrapper."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import pathlib
from typing import List, Optional


class _CAEInferenceMetrics(ctypes.Structure):
    _fields_ = [
        ("morpheme_mask", ctypes.c_uint64),
        ("matched_morphemes", ctypes.c_int32),
        ("centroid_id", ctypes.c_int32),
        ("centroid_hops", ctypes.c_int32),
        ("max_flow", ctypes.c_int32),
        ("path_bottleneck", ctypes.c_int32),
        ("path_length", ctypes.c_int32),
        ("path_nodes", ctypes.c_int32 * 32),
        ("latency_us", ctypes.c_double),
        ("forward_cascade_score", ctypes.c_double),
        ("backward_consistency_score", ctypes.c_double),
        ("anomalies_pruned", ctypes.c_int32),
        ("time_ray_elapsed_us", ctypes.c_double),
        ("estimated_tokens", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class TimeRayMetrics:
    forward_cascade_score: float
    backward_consistency_score: float
    anomalies_pruned: int
    elapsed_us: float


@dataclass(frozen=True)
class AEInferenceOutput:
    query: str
    morpheme_mask_hex: str
    matched_morphemes: int
    centroid_id: int
    centroid_hops: int
    max_flow: int
    path_bottleneck: int
    optimal_path: List[int]
    latency_us: float
    time_ray: TimeRayMetrics
    estimated_tokens: int
    ast_prompt: str
    explainable_trace: str


class AEArchEngine:
    MORPHEME_PREFIX = 0
    MORPHEME_ROOT = 1
    MORPHEME_SUFFIX = 2
    MORPHEME_INFLECTION = 3

    def __init__(
        self,
        rag_nodes: int = 50000,
        rag_avg_degree: int = 6,
        lib_path: Optional[str | pathlib.Path] = None,
    ) -> None:
        self._lib_path = self._resolve_lib_path(lib_path)
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._configure_signatures()
        self._handle = self._lib.ae_engine_create(rag_nodes, rag_avg_degree)
        if not self._handle:
            raise RuntimeError("Failed to initialize AEEngineHandle in C++ core.")

    @staticmethod
    def _resolve_lib_path(custom_path: Optional[str | pathlib.Path]) -> pathlib.Path:
        if custom_path:
            p = pathlib.Path(custom_path).expanduser().resolve()
            if p.exists():
                return p
        here = pathlib.Path(__file__).resolve().parent
        candidates = [
            here.parent / "build" / "libae_mind1.so",
            here / "libae_mind1.so",
            pathlib.Path.cwd() / "build" / "libae_mind1.so",
            pathlib.Path.cwd() / "libae_mind1.so",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError("Could not find libae_mind1.so. Build via CMake or Make first.")

    def _configure_signatures(self) -> None:
        self._lib.ae_engine_create.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self._lib.ae_engine_create.restype = ctypes.c_void_p
        self._lib.ae_engine_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ae_engine_destroy.restype = None
        self._lib.ae_engine_add_morpheme.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_uint8,
            ctypes.c_uint64,
        ]
        self._lib.ae_engine_add_morpheme.restype = None
        self._lib.ae_engine_reroot.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self._lib.ae_engine_reroot.restype = None
        self._lib.ae_engine_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(_CAEInferenceMetrics),
        ]
        self._lib.ae_engine_infer.restype = ctypes.c_int32
        self._lib.ae_engine_get_last_ast.argtypes = [ctypes.c_void_p]
        self._lib.ae_engine_get_last_ast.restype = ctypes.c_char_p
        self._lib.ae_engine_get_last_trace.argtypes = [ctypes.c_void_p]
        self._lib.ae_engine_get_last_trace.restype = ctypes.c_char_p

    def add_morpheme(
        self,
        word: str,
        morpheme_id: int,
        morpheme_type: int = MORPHEME_ROOT,
        compatibility_mask: int = 0xFFFFFFFFFFFFFFF,
    ) -> None:
        self._lib.ae_engine_add_morpheme(
            self._handle,
            word.encode("utf-8"),
            int(morpheme_id),
            int(morpheme_type),
            int(compatibility_mask),
        )

    def reroot(self, new_root: int) -> None:
        self._lib.ae_engine_reroot(self._handle, int(new_root))

    def infer(self, query: str, target_hint: int = -1) -> AEInferenceOutput:
        metrics = _CAEInferenceMetrics()
        rc = self._lib.ae_engine_infer(
            self._handle,
            query.encode("utf-8"),
            int(target_hint),
            ctypes.byref(metrics),
        )
        if rc != 0:
            raise RuntimeError(f"ae_engine_infer failed with status {rc}")

        ast_bytes = self._lib.ae_engine_get_last_ast(self._handle)
        trace_bytes = self._lib.ae_engine_get_last_trace(self._handle)
        path = [int(metrics.path_nodes[i]) for i in range(metrics.path_length)]

        return AEInferenceOutput(
            query=query,
            morpheme_mask_hex=f"0x{metrics.morpheme_mask:X}",
            matched_morphemes=int(metrics.matched_morphemes),
            centroid_id=int(metrics.centroid_id),
            centroid_hops=int(metrics.centroid_hops),
            max_flow=int(metrics.max_flow),
            path_bottleneck=int(metrics.path_bottleneck),
            optimal_path=path,
            latency_us=float(metrics.latency_us),
            time_ray=TimeRayMetrics(
                forward_cascade_score=float(metrics.forward_cascade_score),
                backward_consistency_score=float(metrics.backward_consistency_score),
                anomalies_pruned=int(metrics.anomalies_pruned),
                elapsed_us=float(metrics.time_ray_elapsed_us),
            ),
            estimated_tokens=int(metrics.estimated_tokens),
            ast_prompt=ast_bytes.decode("utf-8", errors="replace") if ast_bytes else "",
            explainable_trace=trace_bytes.decode("utf-8", errors="replace") if trace_bytes else "",
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ae_engine_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "AEArchEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
