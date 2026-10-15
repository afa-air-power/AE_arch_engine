#include "../include/ae_c_api.h"
#include "../lang/humans/parser.hpp"
#include "../include/engine_pipeline.hpp"
#include <../lang/humans/en/parser.hpp>
#include <../lang/humans/ru/parser.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace ae::mind;

    ae::lang::ru::Parser russian_parser;
    ae::lang::en::Parser english_parser;
    ae::lang::Parser* parser_route = ae::lang::select_parser("переподключение");
    auto russian_morphology = russian_parser.parse("переподключение");
    auto routed_morphology = parser_route->parse("переподключение");
    assert(parser_route->language() == ae::lang::Language::RUSSIAN);
    assert(russian_morphology.morphemes.size() >= 3);
    assert(routed_morphology.morphemes.size() == russian_morphology.morphemes.size());
    assert(russian_morphology.morphemes.front().type == ae::lang::MorphemeType::PREFIX);
    assert(russian_morphology.morphemes.front().text == "пере");
    assert(russian_morphology.morphemes.back().text == "ение");
    assert(ae::lang::select_parser("replaying trees")->language() ==
           ae::lang::Language::ENGLISH);
    auto english_morphology = english_parser.parse("replaying cats");
    assert(english_morphology.morphemes.size() == 5);
    assert(english_morphology.morphemes[0].text == "re");
    assert(english_morphology.morphemes[1].text == "play");
    assert(english_morphology.morphemes[2].type == ae::lang::MorphemeType::INFLECTION);
    assert(english_morphology.morphemes[3].text == "cat");
    assert(english_morphology.morphemes[3].end == 13);
    assert(english_morphology.morphemes[4].text == "s");

    AEEnginePipeline pipeline(1000, 6);
    const InferenceResult parsed_inference = pipeline.infer("replaying graph", 42, false);
    assert(parsed_inference.morphology.language == ae::lang::Language::ENGLISH);

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
