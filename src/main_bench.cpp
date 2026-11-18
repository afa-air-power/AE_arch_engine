#include "engine_pipeline.hpp"
#include "lang/en/parser.hpp"
#include "lang/programing/parser.hpp"
#include "lang/ru/parser.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

volatile float g_sink_f = 0.0f;
volatile std::uint64_t g_sink_u64 = 0;

double get_current_rss_mb() {
    std::ifstream statm("/proc/self/statm");
    long size = 0, resident = 0;
    if (statm >> size >> resident) {
        long page_size_kb = ::sysconf(_SC_PAGESIZE) / 1024;
        return static_cast<double>(resident * page_size_kb) / 1024.0;
    }
    return 0.0;
}

template <typename Fn>
double measure_ns_per_call(Fn&& fn, int iterations, int repeats = 5) {
    std::vector<double> runs;
    runs.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            fn(i);
        }
        const auto t1 = Clock::now();
        runs.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations);
    }
    std::sort(runs.begin(), runs.end());
    return runs[runs.size() / 2];
}

double percentile(std::vector<double> samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t idx = static_cast<std::size_t>(
        std::min<double>(static_cast<double>(samples.size() - 1),
                         std::ceil(p * static_cast<double>(samples.size())) - 1.0));
    return samples[idx];
}

void print_line(const char* label, double ns_per_call) {
    std::cout << std::left << std::setw(42) << label << std::right << std::fixed
              << std::setprecision(2) << std::setw(12) << ns_per_call << " ns/call"
              << "   " << std::setw(12)
              << (ns_per_call > 0.0 ? 1e9 / ns_per_call : 0.0) << " ops/s\n";
}

} // namespace

int main() {
    using namespace ae::mind;
    using namespace std::chrono;

    std::cout << "================================================================================\n";
    std::cout << " AE ARCH ENGINE (AE::mind1_v1.1_Stable) — C++23 MODULAR BUILD & BENCHMARK\n";
    std::cout << "================================================================================\n\n";

    float checksum_taylor = 0.0f;
    float checksum_std = 0.0f;

    // ---------------------------------------------------------------------
    // [1] Taylor-Intent Kernel: polynomial activations vs std::tanh
    // ---------------------------------------------------------------------
    TaylorIntentKernel intent_kernel;
    std::array<float, TaylorIntentKernel::INPUT_DIM> sample_input{};
    for (std::size_t i = 0; i < sample_input.size(); ++i) {
        sample_input[i] = -0.8f + 0.05f * static_cast<float>(i);
    }

    constexpr int KERNEL_ITERS = 100000;
    const double taylor_ns = measure_ns_per_call(
        [&](int it) {
            sample_input[0] = static_cast<float>(it & 15) * 0.05f;
            checksum_taylor += intent_kernel.forward_taylor(sample_input)[0];
        },
        KERNEL_ITERS);
    const double std_ns = measure_ns_per_call(
        [&](int it) {
            sample_input[0] = static_cast<float>(it & 15) * 0.05f;
            checksum_std += intent_kernel.forward_transcendental(sample_input)[0];
        },
        KERNEL_ITERS);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[1] Taylor-Intent Kernel (150 neurons, 32->16)\n";
    std::cout << "    forward_taylor (5th-order)               : " << taylor_ns << " ns/call\n";
    std::cout << "    forward_transcendental (std::tanh)       : " << std_ns << " ns/call\n";
    std::cout << "    speedup                                  : " << (std_ns / taylor_ns)
              << "x\n\n";

    // ---------------------------------------------------------------------
    // [2] End-to-end pipeline construction (Mega RAG 50k + centroid index)
    // ---------------------------------------------------------------------
    const auto construct_t0 = Clock::now();
    AEEnginePipeline engine(50000, 6);
    const auto construct_t1 = Clock::now();
    const double construct_ms = duration<double, std::milli>(construct_t1 - construct_t0).count();

    std::cout << "[2] Pipeline cold start\n";
    std::cout << "    AEEnginePipeline(rag_nodes=50000, degree=6) : " << construct_ms
              << " ms (includes mmap CSR + centroid decomposition)\n\n";

    // ---------------------------------------------------------------------
    // [3] Morphemic Patricia Trie: 64-bit compatibility mask compilation
    // ---------------------------------------------------------------------
    const std::array<std::string_view, 4> mask_queries{{
        "переподключение графа мыслей к центроидному дереву памяти",
        "нейросетевая архитектура вычислительного ядра",
        "knowledge graph centroid decomposition",
        "neuro-symbolic reasoning over morphemic trie",
    }};
    const std::array<float, TaylorIntentKernel::INTENT_DIM> zero_intent{};
    const double trie_ns = measure_ns_per_call(
        [&](int it) {
            const auto& q = mask_queries[static_cast<std::size_t>(it) % mask_queries.size()];
            auto [mask, matched] = engine.trie().compile_morpheme_mask(q, zero_intent);
            g_sink_u64 += mask + static_cast<std::uint64_t>(matched);
        },
        200000);

    std::cout << "[3] Morphemic Trie (1804 morphemes / 2580 nodes)\n";
    print_line("    compile_morpheme_mask", trie_ns);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // [4] Human / programming language parsers
    // ---------------------------------------------------------------------
    ae::lang::ru::Parser ru_parser;
    ae::lang::en::Parser en_parser;
    ae::lang::programing::Parser code_parser;

    constexpr std::string_view ru_text =
        "переподключение графа мыслей к центроидному дереву памяти";
    constexpr std::string_view en_text =
        "neuro-symbolic reasoning over morphemic trie topology";
    constexpr std::string_view code_text =
        "int main() { for (int i = 0; i < 10; ++i) { sum += i; } return 0; }";

    const double ru_ns = measure_ns_per_call(
        [&](int) { g_sink_u64 += ru_parser.parse(ru_text).morphemes.size(); }, 20000);
    const double en_ns = measure_ns_per_call(
        [&](int) { g_sink_u64 += en_parser.parse(en_text).morphemes.size(); }, 20000);
    const double code_ns = measure_ns_per_call(
        [&](int) { g_sink_u64 += code_parser.parse(code_text).meaning_tree.children.size(); },
        20000);

    std::cout << "[4] Language parsers\n";
    print_line("    ru::Parser (22 words)", ru_ns);
    print_line("    en::Parser (7 words)", en_ns);
    print_line("    programing::Parser (C-like)", code_ns);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // [5] Mega RAG centroid lookup (mmap CSR, O(depth) climb)
    // ---------------------------------------------------------------------
    MegaRAGCentroidGraph rag(50000, 6);
    rag.build_centroid_index();
    const double centroid_ns = measure_ns_per_call(
        [&](int it) {
            auto [cid, hops] = rag.locate_centroid_cluster(it % rag.n_nodes);
            g_sink_u64 += static_cast<std::uint64_t>(cid + hops);
        },
        1000000);

    std::cout << "[5] Mega RAG (50k nodes, 6 avg degree)\n";
    std::cout << "    CSR mmap footprint                       : " << std::fixed
              << std::setprecision(1)
              << (static_cast<double>(rag.mmap_bytes) / (1024.0 * 1024.0)) << " MB\n";
    print_line("    locate_centroid_cluster", centroid_ns);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // [6] Thought graph: BINUP LCA query + dynamic reroot
    // ---------------------------------------------------------------------
    constexpr int NODES = AEEnginePipeline::DEFAULT_SUBGRAPH_NODES;
    const double lca_ns = measure_ns_per_call(
        [&](int it) {
            auto [lca, weight] =
                engine.thought_graph().query_path_binup(it % NODES, (it * 7 + 13) % NODES);
            g_sink_u64 += static_cast<std::uint64_t>(lca + weight);
        },
        1000000);
    const double reroot_ns = measure_ns_per_call(
        [&](int it) { engine.reroot_thought_graph(1 + it % (NODES - 1)); },
        200000);
    engine.reroot_thought_graph(0); // restore the canonical root for later stages

    std::cout << "[6] Thought Graph (64 nodes, Kruskal DSU + BINUP LCA)\n";
    print_line("    query_path_binup", lca_ns);
    print_line("    reroot (full up/depth rebuild)", reroot_ns);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // [7] Dinitz max-flow decoder (zero-allocation flat arrays)
    // ---------------------------------------------------------------------
    DinitzGraphDecoder decoder(NODES);
    for (const auto& edge : engine.thought_graph().all_edges) {
        decoder.add_static_edge(edge.u, edge.v, edge.weight, edge.tag);
    }
    const double dinitz_ns = measure_ns_per_call(
        [&](int it) {
            const std::uint64_t mask =
                MorphemicPatriciaTrie::BASE_MASK | (static_cast<std::uint64_t>(it & 63) << 28);
            decoder.apply_boolean_mask_and_reset(mask);
            g_sink_u64 += static_cast<std::uint64_t>(decoder.max_flow(0, NODES - 1));
        },
        200000);

    std::cout << "[7] Dinitz Graph Decoder (64 vertices, "
              << decoder.edge_count << " directed edges)\n";
    print_line("    mask reset + max_flow", dinitz_ns);
    std::cout << '\n';

    // ---------------------------------------------------------------------
    // [8] End-to-end infer() latency (morphology -> intent -> flow -> AST)
    // ---------------------------------------------------------------------
    constexpr std::string_view query =
        "переподключение графа мыслей к центроидному дереву памяти и генерация аст токенов";
    const auto result = engine.infer(query, 14158, true);

    constexpr int PIPELINE_ITERS = 2000;
    std::vector<double> sync_us;
    std::vector<double> async_us;
    std::vector<double> core_us;
    sync_us.reserve(PIPELINE_ITERS);
    async_us.reserve(500);
    core_us.reserve(PIPELINE_ITERS);
    for (int i = 0; i < PIPELINE_ITERS; ++i) {
        auto t0 = Clock::now();
        auto r = engine.infer(query, 14158, false);
        auto t1 = Clock::now();
        sync_us.push_back(duration<double, std::micro>(t1 - t0).count());
        core_us.push_back(r.latency_us);
        g_sink_f += r.intent_vector[0];
    }
    for (int i = 0; i < 500; ++i) {
        auto t0 = Clock::now();
        auto r = engine.infer(query, 14158, true);
        auto t1 = Clock::now();
        async_us.push_back(duration<double, std::micro>(t1 - t0).count());
        g_sink_f += r.intent_vector[0];
    }

    std::cout << "[8] End-to-End Pipeline infer()\n";
    std::cout << "    latency_us field (core stages)           : p50 "
              << percentile(core_us, 0.50) << " us / p95 " << percentile(core_us, 0.95)
              << " us\n";
    std::cout << "    wall time, Time-Ray synchronous          : p50 "
              << percentile(sync_us, 0.50) << " us / p95 " << percentile(sync_us, 0.95)
              << " us\n";
    std::cout << "    wall time, Time-Ray std::async           : p50 "
              << percentile(async_us, 0.50) << " us / p95 " << percentile(async_us, 0.95)
              << " us\n\n";

    std::cout << "[9] Result / Memory\n";
    std::cout << "    max_flow                                 : " << result.max_flow << '\n';
    std::cout << "    path_bottleneck                          : " << result.path_bottleneck
              << '\n';
    std::cout << "    AST estimated tokens                     : " << result.ast.estimated_tokens
              << '\n';
    std::cout << "    resident set size                        : " << std::fixed
              << std::setprecision(2) << get_current_rss_mb() << " MB\n\n";
    std::cout << "    Trace: " << result.ast.explainable_trace << "\n\n";
    std::cout << result.ast.ast_text << "\n";

    // Tamper guard: the compiler must not elide the measured work.
    return (checksum_taylor == checksum_std + 1e9f && g_sink_f == 12345.678f) ? 1 : 0;
}
