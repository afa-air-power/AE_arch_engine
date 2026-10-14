#include "../include/ae_c_api.h"
#include "../include/engine_pipeline.hpp"

#include <algorithm>
#include <memory>
#include <string>

struct AEEngineHandle {
    std::unique_ptr<ae::mind1::AEEnginePipeline> pipeline;
    std::string last_ast;
    std::string last_trace;
};

extern "C" {

AEEngineHandle* ae_engine_create(int32_t rag_nodes, int32_t rag_avg_degree) {
    auto* h = new AEEngineHandle();
    h->pipeline = std::make_unique<ae::mind1::AEEnginePipeline>(rag_nodes, rag_avg_degree);
    return h;
}

void ae_engine_destroy(AEEngineHandle* handle) {
    delete handle;
}

void ae_engine_add_morpheme(AEEngineHandle* handle,
                            const char* word,
                            int32_t morpheme_id,
                            uint8_t morpheme_type,
                            uint64_t compatibility_mask) {
    if (!handle || !word) return;
    auto type = static_cast<ae::mind1::MorphemicPatriciaTrie::MorphemeType>(morpheme_type & 0x3);
    handle->pipeline->add_morpheme(word, morpheme_id, type, compatibility_mask);
}

void ae_engine_reroot(AEEngineHandle* handle, int32_t new_root) {
    if (!handle) return;
    handle->pipeline->reroot_thought_graph(new_root);
}

int32_t ae_engine_infer(AEEngineHandle* handle,
                        const char* query_utf8,
                        int32_t target_hint,
                        AEInferenceMetrics* out_metrics) {
    if (!handle || !query_utf8) return -1;
    auto res = handle->pipeline->infer(query_utf8, target_hint, true);
    handle->last_ast = res.ast.ast_text;
    handle->last_trace = res.ast.explainable_trace;

    if (out_metrics) {
        out_metrics->morpheme_mask = res.morpheme_mask;
        out_metrics->matched_morphemes = res.matched_morphemes;
        out_metrics->centroid_id = res.centroid_id;
        out_metrics->centroid_hops = res.centroid_hops;
        out_metrics->max_flow = res.max_flow;
        out_metrics->path_bottleneck = res.path_bottleneck;
        int len = static_cast<int>(std::min<std::size_t>(res.optimal_path.size(), 32));
        out_metrics->path_length = len;
        for (int i = 0; i < len; ++i) {
            out_metrics->path_nodes[i] = res.optimal_path[i];
        }
        out_metrics->latency_us = res.latency_us;
        out_metrics->forward_cascade_score = res.time_ray.forward_cascade_score;
        out_metrics->backward_consistency_score = res.time_ray.backward_consistency_score;
        out_metrics->anomalies_pruned = res.time_ray.anomalies_pruned;
        out_metrics->time_ray_elapsed_us = res.time_ray.elapsed_us;
        out_metrics->estimated_tokens = res.ast.estimated_tokens;
    }
    return 0;
}

const char* ae_engine_get_last_ast(const AEEngineHandle* handle) {
    if (!handle) return "";
    return handle->last_ast.c_str();
}

const char* ae_engine_get_last_trace(const AEEngineHandle* handle) {
    if (!handle) return "";
    return handle->last_trace.c_str();
}

} // extern "C"
