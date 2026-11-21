#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Algorithm-derived feature vector for the trainable graph model
// ----------------------------------------------------------------------------
// Признаки не придумываются заново: они снимаются с уже работающих модулей
// ядра (Taylor-Intent Kernel, морфемный Patricia trie, центроидная
// декомпозиция Mega RAG, Dinitz max-flow, MST/LCA графа мыслей, Time-Ray,
// обратный разбор NL). Модель учится на выходе алгоритмов, а не вместо них.
// ============================================================================

#include "dinitz_decoder.hpp"
#include "mega_rag_centroid.hpp"
#include "morphemic_trie.hpp"
#include "taylor_intent_kernel.hpp"
#include "thought_graph.hpp"
#include "time_ray_ast.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ae::mind {

// Раскладка вектора (118 признаков, все приведены к [0, 1]):
//   [ 0..15] intent-вектор Taylor-Intent Kernel
//   [16..24] морфемная маска и словарь trie
//   [25..27] центроидная декомпозиция Mega RAG
//   [28..31] поток Диница и bottleneck
//   [32..41] топология графа мыслей (путь, LCA-веса, степени)
//   [42..45] Time-Ray сглаживание
//   [46..50] обратный разбор NL (recall/precision/passes)
//   [51..55] метаданные запроса и плотность морфем
//   [56..75] индикаторы 20 семантических кластеров корней
//   [76..95] поддержка кластеров: log1p(число морфем кластера)
//   [96..104] поддержка кластеров у головы предложения (9 профильных)
//   [105..113] максимум поддержки по чанкам-поддеревьям (9 профильных)
//   [114] глубина исходного дерева предложения
//   [115] число RDT-поворотов
//   [116] средняя глубина поднятых голов
//   [117] глубина нормализованного дерева
struct AlgorithmFeatureVector {
    static constexpr int DIM = 118;
    std::array<float, DIM> values{};

    [[nodiscard]] const float* data() const noexcept { return values.data(); }
    [[nodiscard]] float* data() noexcept { return values.data(); }
    [[nodiscard]] float& operator[](std::size_t i) noexcept { return values[i]; }
    [[nodiscard]] float operator[](std::size_t i) const noexcept { return values[i]; }
};

// Девять кластеров, совпадающих с таксономией меток корпуса.
inline constexpr std::array<int, 9> kProfileClusters{0, 1, 2, 3, 4, 7, 8, 9, 19};

inline constexpr std::array<std::string_view, AlgorithmFeatureVector::DIM>
    kAlgorithmFeatureNames{{
        "intent_0", "intent_1", "intent_2", "intent_3",
        "intent_4", "intent_5", "intent_6", "intent_7",
        "intent_8", "intent_9", "intent_10", "intent_11",
        "intent_12", "intent_13", "intent_14", "intent_15",
        "mask_popcount", "matched_morphemes", "mask_domain_bits",
        "mask_gated_bits", "mask_intent_bits", "mask_type_bits",
        "mask_cluster_bits", "trie_dictionary_size", "query_word_count",
        "centroid_id", "centroid_hops", "centroid_depth",
        "max_flow", "path_bottleneck", "bottleneck_flow_ratio", "flow_per_edge",
        "path_length", "path_start", "path_end", "path_mean_node",
        "path_monotonicity", "path_max_edge_weight", "path_min_edge_weight",
        "path_mean_mst_degree", "path_mean_chord_degree", "graph_root",
        "time_ray_forward", "time_ray_backward", "time_ray_anomalies",
        "path_mean_edge_weight", "nl_cluster_recall", "nl_cluster_precision",
        "nl_consistent", "nl_refinement_passes", "nl_reparsed_morphemes",
        "estimated_tokens", "query_bytes", "language_is_russian",
        "language_is_english", "morpheme_density",
        "cluster_graph", "cluster_know", "cluster_flow", "cluster_text",
        "cluster_num", "cluster_time", "cluster_space", "cluster_mind",
        "cluster_compute", "cluster_net", "cluster_phys", "cluster_chem",
        "cluster_bio", "cluster_med", "cluster_econ", "cluster_law",
        "cluster_soc", "cluster_quality", "cluster_nature", "cluster_action",
        "cluster_count_graph", "cluster_count_know", "cluster_count_flow",
        "cluster_count_text", "cluster_count_num", "cluster_count_time",
        "cluster_count_space", "cluster_count_mind", "cluster_count_compute",
        "cluster_count_net", "cluster_count_phys", "cluster_count_chem",
        "cluster_count_bio", "cluster_count_med", "cluster_count_econ",
        "cluster_count_law", "cluster_count_soc", "cluster_count_quality",
        "cluster_count_nature", "cluster_count_action",
        "head_graph", "head_know", "head_flow", "head_text", "head_num",
        "head_mind", "head_compute", "head_net", "head_action",
        "chunk_graph", "chunk_know", "chunk_flow", "chunk_text", "chunk_num",
        "chunk_mind", "chunk_compute", "chunk_net", "chunk_action",
        "sentence_depth", "rdt_rotations", "rdt_head_depth", "rdt_depth",
    }};

namespace feature_detail {

inline float clamp01(double value) noexcept {
    if (!(value > 0.0)) return 0.0f;  // NaN и отрицательные значения
    if (value > 1.0) return 1.0f;
    return static_cast<float>(value);
}

inline double popcount64(std::uint64_t value) noexcept {
    return static_cast<double>(__builtin_popcountll(value));
}

} // namespace feature_detail

// Собирает вектор из сигналов, уже посчитанных алгоритмами ядра.
inline AlgorithmFeatureVector build_algorithm_features(
    const std::array<float, TaylorIntentKernel::INTENT_DIM>& intent,
    std::uint64_t morpheme_mask,
    int matched_morphemes,
    const std::array<std::uint16_t, 20>& cluster_counts,
    const MorphemicPatriciaTrie::SentenceAnalysis& sentence,
    std::size_t dictionary_size,
    std::size_t query_word_count,
    std::size_t query_bytes,
    bool is_russian,
    bool is_english,
    int centroid_id,
    int centroid_hops,
    int centroid_depth,
    int rag_nodes,
    int max_flow,
    int path_bottleneck,
    int directed_edge_count,
    const std::vector<int>& path,
    const ThoughtGraphTopology& tg,
    const TimeRayReport& time_ray,
    double nl_recall,
    double nl_precision,
    bool nl_consistent,
    int nl_passes,
    int nl_reparsed,
    std::size_t estimated_tokens) {
    using feature_detail::clamp01;

    AlgorithmFeatureVector features;
    auto& v = features.values;

    // [0..15] Taylor-Intent Kernel: tanh-выход в [-1, 1] -> [0, 1].
    for (std::size_t i = 0; i < TaylorIntentKernel::INTENT_DIM; ++i) {
        v[i] = clamp01((static_cast<double>(intent[i]) + 1.0) * 0.5);
    }

    // [16..24] морфемная маска (биты 0..7 / 8..15 / 16..23 / 24..27 / 28..47).
    v[16] = clamp01(feature_detail::popcount64(morpheme_mask) / 64.0);
    v[17] = clamp01(static_cast<double>(matched_morphemes) / 32.0);
    v[18] = clamp01(feature_detail::popcount64(morpheme_mask & 0x00000000000000FFULL) / 8.0);
    v[19] = clamp01(feature_detail::popcount64(morpheme_mask & 0x000000000000FF00ULL) / 8.0);
    v[20] = clamp01(feature_detail::popcount64(morpheme_mask & 0x0000000000FF0000ULL) / 8.0);
    v[21] = clamp01(feature_detail::popcount64(morpheme_mask & 0x000000000F000000ULL) / 4.0);
    v[22] = clamp01(feature_detail::popcount64(morpheme_mask & 0x0000FFFFF0000000ULL) / 20.0);
    v[23] = clamp01(static_cast<double>(dictionary_size) / 4096.0);
    v[24] = clamp01(static_cast<double>(query_word_count) / 32.0);

    // [25..27] центроидная декомпозиция.
    v[25] = clamp01(static_cast<double>(centroid_id) / std::max(rag_nodes, 1));
    v[26] = clamp01(static_cast<double>(centroid_hops) / 8.0);
    v[27] = clamp01(static_cast<double>(centroid_depth) / 16.0);

    // [28..31] поток Диница.
    const int edge_pairs = std::max(directed_edge_count / 2, 1);
    v[28] = clamp01(static_cast<double>(max_flow) / 256.0);
    v[29] = clamp01(static_cast<double>(path_bottleneck) / 256.0);
    v[30] = clamp01(static_cast<double>(path_bottleneck) /
                    static_cast<double>(std::max(max_flow, 1)));
    v[31] = clamp01(static_cast<double>(max_flow) /
                    (static_cast<double>(edge_pairs) * 16.0));

    // [32..41] топология графа мыслей.
    v[32] = clamp01(static_cast<double>(path.size()) / 32.0);
    double edge_sum = 0.0;
    double edge_count = 0.0;
    if (!path.empty()) {
        double sum = 0.0;
        int increasing = 0;
        int max_w = 0;
        int min_w = ThoughtGraphTopology::kInfWeight;
        double mst_degree = 0.0;
        double chord_degree = 0.0;
        for (std::size_t i = 0; i < path.size(); ++i) {
            const int node = path[i];
            sum += static_cast<double>(node);
            if (node >= 0 && node < tg.n) {
                mst_degree += static_cast<double>(tg.adj_mst[node].size());
                chord_degree += static_cast<double>(tg.adj_horiz[node].size());
            }
            if (i > 0) {
                if (path[i] > path[i - 1]) ++increasing;
                const auto [lca_max, max_weight] = tg.query_path_binup(path[i - 1], path[i]);
                const auto [lca_min, min_weight] = tg.query_path_min_binup(path[i - 1], path[i]);
                (void)lca_max;
                (void)lca_min;
                max_w = std::max(max_w, max_weight);
                min_w = std::min(min_w, min_weight);
                edge_sum += static_cast<double>(max_weight);
                edge_count += 1.0;
            }
        }
        const double count = static_cast<double>(path.size());
        v[33] = clamp01(static_cast<double>(path.front()) / 64.0);
        v[34] = clamp01(static_cast<double>(path.back()) / 64.0);
        v[35] = clamp01(sum / (count * 64.0));
        v[36] = path.size() > 1
                    ? clamp01(static_cast<double>(increasing) / (count - 1.0))
                    : 0.0f;
        v[37] = clamp01(static_cast<double>(max_w) / 256.0);
        v[38] = min_w == ThoughtGraphTopology::kInfWeight
                    ? 1.0f
                    : clamp01(static_cast<double>(min_w) / 256.0);
        v[39] = clamp01(mst_degree / (count * 8.0));
        v[40] = clamp01(chord_degree / (count * 8.0));
    } else {
        v[38] = 1.0f;
    }
    v[41] = clamp01(static_cast<double>(tg.current_root) / 64.0);

    // [42..45] Time-Ray.
    v[42] = clamp01(time_ray.forward_cascade_score /
                    (1.0 + time_ray.forward_cascade_score));
    v[43] = clamp01(time_ray.backward_consistency_score);
    v[44] = clamp01(static_cast<double>(time_ray.anomalies_pruned) / 8.0);
    // Детерминированная замена wall-clock времени: средний вес ребра пути.
    v[45] = edge_count > 0.0 ? clamp01(edge_sum / (edge_count * 256.0)) : 0.0f;

    // [46..50] обратный разбор NL.
    v[46] = clamp01(nl_recall);
    v[47] = clamp01(nl_precision);
    v[48] = nl_consistent ? 1.0f : 0.0f;
    v[49] = clamp01(static_cast<double>(nl_passes) / 4.0);
    v[50] = clamp01(static_cast<double>(nl_reparsed) / 64.0);

    // [51..55] метаданные запроса.
    v[51] = clamp01(static_cast<double>(estimated_tokens) / 2048.0);
    v[52] = clamp01(static_cast<double>(query_bytes) / 512.0);
    v[53] = is_russian ? 1.0f : 0.0f;
    v[54] = is_english ? 1.0f : 0.0f;
    v[55] = clamp01(static_cast<double>(matched_morphemes) /
                    static_cast<double>(std::max<std::size_t>(query_word_count, 1)));

    // [56..75] индикаторы семантических кластеров из маски trie (биты 28..47).
    for (int cluster = 0; cluster < 20; ++cluster) {
        const std::uint64_t bit = 1ULL << (28 + cluster);
        v[56 + static_cast<std::size_t>(cluster)] =
            (morpheme_mask & bit) ? 1.0f : 0.0f;
    }

    // [76..95] поддержка кластеров: log1p(число морфем кластера).
    // Маска отвечает на вопрос "тема встретилась", счётчик — "насколько
    // сильно". Это различает запросы с одинаковым набором битов.
    const double count_scale = 1.0 / std::log1p(16.0);
    for (int cluster = 0; cluster < 20; ++cluster) {
        const double count = static_cast<double>(cluster_counts[cluster]);
        v[76 + static_cast<std::size_t>(cluster)] =
            clamp01(std::log1p(count) * count_scale);
    }

    // [96..104] поддержка у головы предложения. Голова — это последнее
    // содержательное слово чанка; для известных составных терминов она
    // получает поддержку главного кластера фразы.
    // [105..113] максимум по чанкам: показывает локально связные поддеревья,
    // даже если в целом предложение про другую тему.
    for (std::size_t index = 0; index < kProfileClusters.size(); ++index) {
        const std::size_t cluster =
            static_cast<std::size_t>(kProfileClusters[index]);
        const double head_count = static_cast<double>(sentence.head_counts[cluster]);
        const double chunk_count =
            static_cast<double>(sentence.chunk_max_counts[cluster]);
        v[96 + index] = clamp01(std::log1p(head_count) * count_scale);
        v[105 + index] = clamp01(std::log1p(chunk_count) * count_scale);
    }

    // [114] глубина дерева: 2 — одна голова, 3 — есть зависимости.
    v[114] = clamp01(static_cast<double>(sentence.max_depth) / 8.0);

    // [115..117] RDT-представление. Поворот — наблюдаемая структурная
    // операция, а не скрытая эвристика. Она показывает, сколько голов было
    // поднято над контейнером и насколько изменилась глубина дерева.
    v[115] = clamp01(static_cast<double>(sentence.rotations.size()) / 16.0);
    double lifted_depth = 0.0;
    for (const auto& rotation : sentence.rotations) {
        lifted_depth += static_cast<double>(rotation.after_depth);
    }
    v[116] = sentence.rotations.empty()
        ? 0.0f
        : clamp01(lifted_depth /
                  (static_cast<double>(sentence.rotations.size()) * 8.0));
    int normalized_depth = 0;
    for (const auto& node : sentence.nodes) {
        normalized_depth = std::max(normalized_depth, node.normalized_depth);
    }
    v[117] = clamp01(static_cast<double>(normalized_depth) / 8.0);

    return features;
}

} // namespace ae::mind
