#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Unified End-to-End Inference Pipeline
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include "algorithm_features.hpp"
#include "dinitz_decoder.hpp"
#include "lang/parser.hpp"
#include "mega_rag_centroid.hpp"
#include "morphemic_trie.hpp"
#include "natural_language_output.hpp"
#include "taylor_intent_kernel.hpp"
#include "thought_graph.hpp"
#include "time_ray_ast.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ae::mind {

struct InferenceResult {
    std::array<float, TaylorIntentKernel::INTENT_DIM> intent_vector{};
    lang::ParseResult morphology;
    std::uint64_t morpheme_mask = 0;
    int matched_morphemes = 0;
    int centroid_id = 0;
    int centroid_hops = 0;
    int max_flow = 0;
    int path_bottleneck = 0;
    std::vector<int> optimal_path;
    TimeRayReport time_ray;
    ASTPromptResult ast;
    NaturalLanguageReport natural_language;
    double latency_us = 0.0;
};

// Сигналы, которые считают алгоритмы ядра до AST/NL-стадии. Используются и
// конвейером вывода, и извлечением признаков для обучаемой модели, поэтому
// обучение и runtime видят одно и то же представление.
struct CoreSignals {
    lang::ParseResult morphology;
    std::array<float, TaylorIntentKernel::INTENT_DIM> intent{};
    std::uint64_t morpheme_mask = 0;
    int matched_morphemes = 0;
    int centroid_id = 0;
    int centroid_hops = 0;
    int max_flow = 0;
    int path_bottleneck = 0;
    std::vector<int> optimal_path;
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

    // Признаки, снятые с алгоритмов ядра (см. algorithm_features.hpp).
    [[nodiscard]] AlgorithmFeatureVector extract_features(std::string_view query,
                                                          int target_hint = -1);

    [[nodiscard]] const ThoughtGraphTopology& thought_graph() const noexcept {
        return thought_graph_;
    }

    [[nodiscard]] const MorphemicPatriciaTrie& trie() const noexcept {
        return trie_;
    }

    [[nodiscard]] const MegaRAGCentroidGraph& rag() const noexcept {
        return rag_;
    }

    [[nodiscard]] DinitzGraphDecoder& decoder() noexcept {
        return decoder_;
    }

    [[nodiscard]] const DinitzGraphDecoder& decoder() const noexcept {
        return decoder_;
    }

    InferenceResult infer(std::string_view query,
                          int target_hint = -1,
                          bool run_time_ray_async = true);

private:
    [[nodiscard]] CoreSignals compute_core_signals(std::string_view query,
                                                   int target_hint);

    TaylorIntentKernel intent_kernel_;
    MorphemicPatriciaTrie trie_;
    MegaRAGCentroidGraph rag_;
    ThoughtGraphTopology thought_graph_;
    DinitzGraphDecoder decoder_;
    int sink_node_ = DEFAULT_SUBGRAPH_NODES - 1;
};

} // namespace ae::mind
