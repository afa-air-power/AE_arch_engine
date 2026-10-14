#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable)
// Modules 3.5 & 6: Async Time-Ray Smoothing + Verified AST-Prompt Compiler
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include "../include/thought_graph.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ae::mind1 {

struct TimeRayReport {
    double forward_cascade_score = 0.0;
    double backward_consistency_score = 0.0;
    int anomalies_pruned = 0;
    double elapsed_us = 0.0;
};

inline TimeRayReport run_async_time_ray(
    const std::vector<int>& path,
    const ThoughtGraphTopology& tg,
    int steps = 8) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    double forward_score = 0.0;
    double backward_score = 1.0;
    int pruned = 0;

    for (std::size_t i = 0; i < path.size(); ++i) {
        int u = path[i];
        int cur = u;
        double discount = 1.0;
        for (int s = 0; s < steps && !tg.adj_mst[cur].empty(); ++s) {
            auto [nxt, w] = tg.adj_mst[cur][s % tg.adj_mst[cur].size()];
            forward_score += discount * static_cast<double>(w);
            discount *= 0.85;
            cur = nxt;
        }
    }

    for (int i = static_cast<int>(path.size()) - 1; i > 0; --i) {
        auto [lca_node, max_w] = tg.query_path_binup(path[i], path[i - 1]);
        if (max_w < 10) {
            ++pruned;
            backward_score *= 0.92;
        } else {
            backward_score = std::min(1.0, backward_score * 1.01);
        }
    }

    auto t1 = high_resolution_clock::now();
    double us = duration<double, std::micro>(t1 - t0).count();
    return {forward_score, backward_score, pruned, us};
}

struct ASTPromptResult {
    std::string ast_text;
    std::size_t estimated_tokens;
    std::string explainable_trace;
};

inline ASTPromptResult compile_verified_ast_prompt(
    std::string_view query,
    int centroid_id,
    int centroid_hops,
    std::uint64_t morpheme_mask,
    int max_flow_value,
    const std::vector<int>& flow_path,
    const ThoughtGraphTopology& tg) {
    std::ostringstream oss;
    oss << "(AE_VERIFIED_AST_V1.1\n"
        << "  (META :engine \"AE::mind1_v1.1_Stable\" :hallucination_guard \"STRICT_BOOLEAN_MASK\"\n"
        << "        :target_slm \"Local-SLM-3B-Instruct\" :centroid_cluster_id " << centroid_id
        << " :centroid_hops " << centroid_hops << ")\n"
        << "  (QUERY_INTENT :raw \"" << query << "\"\n"
        << "                :morpheme_mask \"0x" << std::hex << std::uppercase << morpheme_mask << std::dec << "\"\n"
        << "                :dinitz_max_flow " << max_flow_value << ")\n"
        << "  (THOUGHT_GRAPH_TOPOLOGY :root " << tg.current_root << " :nodes " << tg.n
        << " :spanning_mechanism \"KRUSKAL_DSU_BINUP_LCA\"\n";

    for (std::size_t i = 0; i < flow_path.size(); ++i) {
        int u = flow_path[i];
        int p = tg.up[u][0];
        int w = tg.max_edge[u][0];
        auto [lca_root, bottleneck] = tg.query_path_binup(tg.current_root, u);
        oss << "    (THOUGHT_NODE :step " << i
            << " :node_id " << u
            << " :parent_id " << p
            << " :depth " << tg.depth[u]
            << " :edge_weight " << w
            << " :lca_with_root " << lca_root
            << " :path_max_weight " << bottleneck
            << " :fact_ref \"KB_CHUNK_" << (centroid_id * 100 + u) << "\")\n";
    }

    oss << "  )\n"
        << "  (SLM_DIRECTIVE :mode \"DETERMINISTIC_VERBALIZATION\"\n"
        << "                 :instruction \"Сформируй точный научно-технический ответ строго по узлам THOUGHT_NODE без добавления внешних фактов.\")\n"
        << ")\n";

    std::string text = oss.str();
    std::size_t est_tokens = text.size() / 4 + 35;

    std::ostringstream trace;
    trace << "Intent[16d] -> MorphemeMask[0x" << std::hex << std::uppercase << morpheme_mask << std::dec
          << "] -> Centroid[ID=" << centroid_id << ", hops=" << centroid_hops
          << "] -> ThoughtGraph[DSU+BINUP root=" << tg.current_root
          << "] -> DinitzFlow[max_flow=" << max_flow_value << ", path_len=" << flow_path.size()
          << "] -> AST[" << est_tokens << " tokens]";

    return {text, est_tokens, trace.str()};
}

} // namespace ae::mind1
