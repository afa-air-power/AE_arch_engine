#include "../include/engine_pipeline.hpp"

#include <chrono>
#include <future>
#include <random>

namespace ae::mind1 {

AEEnginePipeline::AEEnginePipeline(int rag_nodes, int rag_avg_degree)
    : intent_kernel_(42),
      trie_(),
      rag_(rag_nodes, rag_avg_degree),
      thought_graph_(),
      decoder_(DEFAULT_SUBGRAPH_NODES) {
    rag_.build_centroid_index();

    std::vector<ThoughtGraphTopology::Edge> context_edges;
    std::mt19937 sub_rng(2026);
    for (int i = 1; i < DEFAULT_SUBGRAPH_NODES; ++i) {
        int p = (i < 4) ? 0 : static_cast<int>(sub_rng() % i);
        int w = 40 + static_cast<int>(sub_rng() % 60);
        std::uint8_t tag = static_cast<std::uint8_t>(sub_rng() % 16);
        context_edges.push_back({p, i, w, static_cast<int>(context_edges.size()), tag});
    }
    for (int h = 0; h < 24; ++h) {
        int u = 1 + (sub_rng() % (DEFAULT_SUBGRAPH_NODES / 2));
        int v = (DEFAULT_SUBGRAPH_NODES / 2) + (sub_rng() % (DEFAULT_SUBGRAPH_NODES / 2));
        int w = 25 + static_cast<int>(sub_rng() % 50);
        std::uint8_t tag = static_cast<std::uint8_t>(sub_rng() % 16);
        context_edges.push_back({u, v, w, static_cast<int>(context_edges.size()), tag});
    }

    thought_graph_.init_from_subgraph(DEFAULT_SUBGRAPH_NODES, context_edges);

    for (const auto& e : context_edges) {
        decoder_.add_static_edge(e.u, e.v, e.weight, e.tag);
    }
    int mid1 = 8, mid2 = 21, mid3 = 38, mid4 = 52;
    sink_node_ = DEFAULT_SUBGRAPH_NODES - 1;
    decoder_.add_static_edge(0, mid1, 120, 3);
    decoder_.add_static_edge(mid1, mid2, 115, 4);
    decoder_.add_static_edge(mid2, mid3, 110, 5);
    decoder_.add_static_edge(mid3, mid4, 105, 6);
    decoder_.add_static_edge(mid4, sink_node_, 100, 7);
}

void AEEnginePipeline::add_morpheme(std::string_view word, int id,
                                    MorphemicPatriciaTrie::MorphemeType type,
                                    std::uint64_t compatibility_mask) {
    trie_.insert(word, id, type, compatibility_mask);
}

void AEEnginePipeline::reroot_thought_graph(int new_root) noexcept {
    thought_graph_.reroot(new_root);
}

InferenceResult AEEnginePipeline::infer(std::string_view query,
                                        int target_hint,
                                        bool run_time_ray_async) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();

    InferenceResult res{};
    std::array<float, TaylorIntentKernel::INPUT_DIM> features{};
    for (std::size_t i = 0; i < features.size(); ++i) {
        char c = query.empty() ? 0 : query[i % query.size()];
        features[i] = static_cast<float>(static_cast<unsigned char>(c)) / 255.0f - 0.5f;
    }

    res.intent_vector = intent_kernel_.forward_taylor(features);
    auto [mask, m_cnt] = trie_.compile_morpheme_mask(query, res.intent_vector);
    res.morpheme_mask = mask;
    res.matched_morphemes = m_cnt;

    if (target_hint < 0) {
        std::uint64_t hash_val = mask ^ static_cast<std::uint64_t>(query.size() * 1315423911ULL);
        target_hint = static_cast<int>(hash_val % static_cast<std::uint64_t>(rag_.n_nodes));
    }
    auto [cid, hops] = rag_.locate_centroid_cluster(target_hint);
    res.centroid_id = cid;
    res.centroid_hops = hops;

    decoder_.apply_boolean_mask_and_reset(mask);
    res.max_flow = decoder_.max_flow(0, sink_node_);
    decoder_.extract_optimal_thought_path(0, sink_node_, res.optimal_path);

    int bottleneck = 0;
    for (std::size_t i = 1; i < res.optimal_path.size(); ++i) {
        auto [lca_u, max_w] = thought_graph_.query_path_binup(
            res.optimal_path[i - 1], res.optimal_path[i]);
        bottleneck = std::max(bottleneck, max_w);
    }
    res.path_bottleneck = bottleneck;

    auto t1 = high_resolution_clock::now();
    res.latency_us = duration<double, std::micro>(t1 - t0).count();

    if (run_time_ray_async) {
        auto fut = std::async(std::launch::async, [&]() {
            return run_async_time_ray(res.optimal_path, thought_graph_, 8);
        });
        res.time_ray = fut.get();
    } else {
        res.time_ray = run_async_time_ray(res.optimal_path, thought_graph_, 8);
    }

    res.ast = compile_verified_ast_prompt(
        query, res.centroid_id, res.centroid_hops,
        res.morpheme_mask, res.max_flow, res.optimal_path, thought_graph_);

    return res;
}

} // namespace ae::mind1
