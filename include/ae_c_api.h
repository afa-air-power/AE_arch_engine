#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable) — Pure C ABI for Python/Rust/Go FFI
// Author: Andrey Firstov (IU5 BMSTU)
// ADR: docs/adr/0011-c-abi-and-python-bindings.md
// ============================================================================

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
// Непрозрачный handle: клиент не видит C++-типы и не управляет памятью внутри.
typedef struct AEEngineHandle AEEngineHandle;

// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
// POD-метрики фиксированного размера; поля должны совпадать с ae_engine.py.
typedef struct AEInferenceMetrics {
    uint64_t morpheme_mask;
    int32_t matched_morphemes;
    int32_t centroid_id;
    int32_t centroid_hops;
    int32_t max_flow;
    int32_t path_bottleneck;
    int32_t path_length;
    int32_t path_nodes[32];
    double latency_us;
    double forward_cascade_score;
    double backward_consistency_score;
    int32_t anomalies_pruned;
    double time_ray_elapsed_us;
    size_t estimated_tokens;

    /* Natural-language round-trip (must stay in sync with ae_engine.py). */
    double nl_cluster_recall;
    double nl_cluster_precision;
    int32_t nl_reparsed_morphemes;
    int32_t nl_refinement_passes;
    int32_t nl_consistent;
} AEInferenceMetrics;

// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
AEEngineHandle* ae_engine_create(int32_t rag_nodes, int32_t rag_avg_degree);
// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
void ae_engine_destroy(AEEngineHandle* handle);

// ADR-0002: docs/adr/0002-morphemic-patricia-trie.md
// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
void ae_engine_add_morpheme(AEEngineHandle* handle,
                            const char* word,
                            int32_t morpheme_id,
                            uint8_t morpheme_type,
                            uint64_t compatibility_mask);

// ADR-0005: docs/adr/0005-thought-graph-mst-lca.md
void ae_engine_reroot(AEEngineHandle* handle, int32_t new_root);

// ADR-0010: docs/adr/0010-unified-pipeline.md
// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
int32_t ae_engine_infer(AEEngineHandle* handle,
                        const char* query_utf8,
                        int32_t target_hint,
                        AEInferenceMetrics* out_metrics);

// ADR-0007: docs/adr/0007-verified-ast-prompt.md
// ADR-0011: docs/adr/0011-c-abi-and-python-bindings.md
// Возвращают const char*, принадлежащий handle; живут до следующего infer/destroy.
const char* ae_engine_get_last_ast(const AEEngineHandle* handle);
const char* ae_engine_get_last_trace(const AEEngineHandle* handle);
// ADR-0008: docs/adr/0008-natural-language-round-trip.md
const char* ae_engine_get_last_nl(const AEEngineHandle* handle);
const char* ae_engine_get_last_nl_feedback(const AEEngineHandle* handle);

#ifdef __cplusplus
}
#endif
