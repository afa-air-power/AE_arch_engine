#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable)
// Unified End-to-End Inference Pipeline
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include "../include/dinitz_decoder.hpp"
#include "../include/mega_rag_centroid.hpp"
#include "../include/morphemic_trie.hpp"
#include "../include/taylor_intent_kernel.hpp"
#include "../include/thought_graph.hpp"
#include "../include/time_ray_ast.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ae::mind1 {

struct InferenceResult {
    std::array<float, TaylorIntentKernel::INTENT_DIM> intent_vector{};
    std::uint64_t morpheme_mask = 0;
    int matched_morphemes = 0;
    int centroid_id = 0;
    int centroid_hops = 0;
    int max_flow = 0;
    int path_bottleneck = 0;
    std::vector<int> optimal_path;
    TimeRayReport time_ray;
    ASTPromptResult ast;
    double latency_us = 0.0;
};

class AEEnginePipeline {
public:
    static constexpr int DEFAULT_SUBGRAPH_NODES = 64;

    explicit AEEnginePipeline(int rag_nodes = 100000, int rag_avg_degree = 6);

    void add_morpheme(std::string_view word, int id,
                      MorphemicPatriciaTrie::MorphemeType type,
                      std::uint64_t compatibility_mask);

    void reroot_thought_graph(int new_root) noexcept;

    InferenceResult infer(std::string_view query,
                          int target_hint = -1,
                          bool run_time_ray_async = true);

private:
    TaylorIntentKernel intent_kernel_;
    MorphemicPatriciaTrie trie_;
    MegaRAGCentroidGraph rag_;
    ThoughtGraphTopology thought_graph_;
    DinitzGraphDecoder decoder_;
    int sink_node_ = DEFAULT_SUBGRAPH_NODES - 1;
};

} // namespace ae::mind1
