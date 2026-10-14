#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable) — Pure C ABI for Python/Rust/Go FFI
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AEEngineHandle AEEngineHandle;

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
} AEInferenceMetrics;

AEEngineHandle* ae_engine_create(int32_t rag_nodes, int32_t rag_avg_degree);
void ae_engine_destroy(AEEngineHandle* handle);

void ae_engine_add_morpheme(AEEngineHandle* handle,
                            const char* word,
                            int32_t morpheme_id,
                            uint8_t morpheme_type,
                            uint64_t compatibility_mask);

void ae_engine_reroot(AEEngineHandle* handle, int32_t new_root);

int32_t ae_engine_infer(AEEngineHandle* handle,
                        const char* query_utf8,
                        int32_t target_hint,
                        AEInferenceMetrics* out_metrics);

const char* ae_engine_get_last_ast(const AEEngineHandle* handle);
const char* ae_engine_get_last_trace(const AEEngineHandle* handle);

#ifdef __cplusplus
}
#endif
