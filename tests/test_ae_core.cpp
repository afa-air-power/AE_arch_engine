#include "../include/ae_c_api.h"
#include "../include/engine_pipeline.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace ae::mind1;

    TaylorIntentKernel kernel;
    std::array<float, TaylorIntentKernel::INPUT_DIM> x{};
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = -0.5f + 0.03f * static_cast<float>(i);
    auto y_taylor = kernel.forward_taylor(x);
    auto y_std = kernel.forward_transcendental(x);
    for (std::size_t i = 0; i < y_taylor.size(); ++i) {
        assert(std::abs(y_taylor[i] - y_std[i]) < 0.01f);
    }

    MorphemicPatriciaTrie trie;
    auto [mask, count] = trie.compile_morpheme_mask("переподключение графа мыслей", y_taylor);
    assert(count >= 3);
    assert(mask != 0);

    MegaRAGCentroidGraph rag(5000, 6);
    rag.build_centroid_index();
    auto [cid, hops] = rag.locate_centroid_cluster(1234);
    assert(cid >= 0 && cid < 5000);
    assert(hops >= 1 && hops <= 4);

    AEEngineHandle* handle = ae_engine_create(10000, 6);
    assert(handle != nullptr);
    AEInferenceMetrics metrics{};
    int rc = ae_engine_infer(handle, "граф мыслей и центроид памяти", 42, &metrics);
    assert(rc == 0);
    assert(metrics.max_flow > 0);
    assert(metrics.path_length >= 2);
    assert(metrics.estimated_tokens > 100);
    std::string ast = ae_engine_get_last_ast(handle);
    assert(ast.find("AE_VERIFIED_AST_V1.1") != std::string::npos);
    ae_engine_destroy(handle);

    std::cout << "[PASS] All C++23 AE::mind1 unit and C-ABI tests succeeded.\n";
    return 0;
}
