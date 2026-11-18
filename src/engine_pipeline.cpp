#include "engine_pipeline.hpp"

#include <chrono>
#include <future>
#include <random>

namespace ae::mind {

AEEnginePipeline::AEEnginePipeline(int rag_nodes, int rag_avg_degree)
    : intent_kernel_(42),
      trie_(),
      rag_(rag_nodes, rag_avg_degree),
      thought_graph_(),
      decoder_(DEFAULT_SUBGRAPH_NODES) {
    rag_.build_centroid_index();

    // Единый источник топологии: граф мыслей и сеть Диница строятся из одного
    // набора рёбер, поэтому цепочка, которую печатает AST, является настоящим
    // путём в остовном дереве (совпадают parent_id / depth / LCA).
    std::vector<ThoughtGraphTopology::Edge> topology;
    std::mt19937 sub_rng(2026);

    // 1. Семантические цепочки root -> sink. Инвариантная цепочка размечена
    //    тегами 0..4 (они всегда входят в BASE_MASK) и гарантирует, что путь
    //    существует при любой булевой маске. Остальные цепочки размечены
    //    тегами 28 + cluster, поэтому открываются только теми морфемными
    //    кластерами, которые реально встретились в запросе.
    struct ChainSpec {
        std::array<int, 6> nodes;
        MorphemicPatriciaTrie::Semantic cluster;
        bool invariant;
        int capacity;
        int step;
    };

    constexpr int sink = DEFAULT_SUBGRAPH_NODES - 1;
    const std::array<ChainSpec, 10> chains{{
        {{0, 8, 21, 38, 52, sink}, MorphemicPatriciaTrie::Semantic::GRAPH, true, 200, 20},
        {{0, 3, 17, 29, 47, sink}, MorphemicPatriciaTrie::Semantic::KNOW, false, 90, 10},
        {{0, 6, 24, 35, 49, sink}, MorphemicPatriciaTrie::Semantic::FLOW, false, 80, 10},
        {{0, 12, 27, 41, 56, sink}, MorphemicPatriciaTrie::Semantic::TEXT, false, 70, 10},
        {{0, 5, 19, 33, 45, sink}, MorphemicPatriciaTrie::Semantic::NUM, false, 60, 10},
        {{0, 10, 25, 37, 51, sink}, MorphemicPatriciaTrie::Semantic::COMPUTE, false, 55, 8},
        {{0, 14, 26, 39, 54, sink}, MorphemicPatriciaTrie::Semantic::MIND, false, 50, 8},
        {{0, 2, 16, 31, 44, sink}, MorphemicPatriciaTrie::Semantic::GRAPH, false, 45, 6},
        {{0, 7, 22, 34, 48, sink}, MorphemicPatriciaTrie::Semantic::NET, false, 40, 6},
        {{0, 11, 28, 40, 53, sink}, MorphemicPatriciaTrie::Semantic::ACTION, false, 35, 5},
    }};

    std::array<bool, DEFAULT_SUBGRAPH_NODES> used{};
    used[0] = true;
    used[sink] = true;
    for (const auto& chain : chains) {
        for (int node : chain.nodes) used[node] = true;
        int capacity = chain.capacity;
        for (std::size_t i = 0; i + 1 < chain.nodes.size(); ++i) {
            std::uint8_t tag = chain.invariant
                ? static_cast<std::uint8_t>(i)
                : static_cast<std::uint8_t>(28 + static_cast<int>(chain.cluster));
            topology.push_back({chain.nodes[i], chain.nodes[i + 1], capacity,
                                static_cast<int>(topology.size()), tag});
            capacity = std::max(1, capacity - chain.step);
        }
    }

    // 2. Ветви: оставшиеся вершины цепляются к более ранним. Тег — кластер,
    //    поэтому проводимость ветвей тоже зависит от морфемной маски.
    for (int i = 1; i < DEFAULT_SUBGRAPH_NODES; ++i) {
        if (used[i]) continue;
        int parent = static_cast<int>(sub_rng() % static_cast<unsigned>(i));
        int w = 40 + static_cast<int>(sub_rng() % 60);
        std::uint8_t tag = static_cast<std::uint8_t>(28 + sub_rng() % 20);
        topology.push_back({parent, i, w, static_cast<int>(topology.size()), tag});
    }

    // 3. Хорды создают циклы: в остовное дерево они не входят, но участвуют
    //    в максимальном потоке, поэтому max_flow зависит от маски запроса.
    for (int h = 0; h < 24; ++h) {
        int u = 1 + static_cast<int>(sub_rng() % (DEFAULT_SUBGRAPH_NODES / 2));
        int v = (DEFAULT_SUBGRAPH_NODES / 2) +
                static_cast<int>(sub_rng() % (DEFAULT_SUBGRAPH_NODES / 2));
        int w = 25 + static_cast<int>(sub_rng() % 50);
        std::uint8_t tag = static_cast<std::uint8_t>(28 + sub_rng() % 20);
        topology.push_back({u, v, w, static_cast<int>(topology.size()), tag});
    }

    thought_graph_.init_from_subgraph(DEFAULT_SUBGRAPH_NODES, topology);

    for (const auto& edge : topology) {
        decoder_.add_static_edge(edge.u, edge.v, edge.weight, edge.tag);
    }
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
    res.morphology = lang::select_parser(query)->parse(query);
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

    // Верифицированная цепочка — единственный путь root -> sink в остовном
    // дереве, то есть ровно те рёбра, по которым строятся parent_id / depth /
    // LCA в AST-промпте.
    res.optimal_path = thought_graph_.path_between(0, sink_node_);

    int bottleneck = ThoughtGraphTopology::kInfWeight;
    for (std::size_t i = 1; i < res.optimal_path.size(); ++i) {
        auto [lca_node, min_w] = thought_graph_.query_path_min_binup(
            res.optimal_path[i - 1], res.optimal_path[i]);
        (void)lca_node;
        bottleneck = std::min(bottleneck, min_w);
    }
    res.path_bottleneck = bottleneck == ThoughtGraphTopology::kInfWeight ? 0 : bottleneck;

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

} // namespace ae::mind
