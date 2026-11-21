#include "ae_c_api.h"
#include "engine_pipeline.hpp"
#include "lang/en/parser.hpp"
#include "lang/interface.hpp"
#include "lang/parser.hpp"
#include "lang/programing/parser.hpp"
#include "lang/ru/parser.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace ae::mind;

    ae::lang::ru::Parser russian_parser;
    ae::lang::en::Parser english_parser;
    ae::lang::Parser* parser_route = ae::lang::select_parser("переподключение");
    ae::lang::human::Interface* human_interface =
        ae::lang::select_parser(ae::lang::LanguageFamily::HUMAN, "переподключение");
    auto russian_morphology = russian_parser.parse("переподключение");
    auto routed_morphology = parser_route->parse("переподключение");
    const auto neuro_morphology =
        russian_parser.parse("нейрокомпьютерная архитектура распределённого графа");
    assert(neuro_morphology.morphemes.size() >= 8);
    assert(neuro_morphology.morphemes.front().text == "нейро");
    assert(neuro_morphology.morphemes.front().type == ae::lang::MorphemeType::PREFIX);
    assert(human_interface != nullptr);
    assert(human_interface->family() == ae::lang::LanguageFamily::HUMAN);
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

    ae::lang::programing::Parser programming_parser;
    ae::lang::LanguageInterface* language_interface = &programming_parser;
    const auto* selected_programming_parser = ae::lang::select_parser(
        ae::lang::LanguageFamily::PROGRAMMING, "int main() {}");
    assert(selected_programming_parser != nullptr);
    assert(selected_programming_parser->family() ==
           ae::lang::LanguageFamily::PROGRAMMING);
    assert(ae::lang::select_parser(
               ae::lang::LanguageFamily::SCIENCE, "E = mc^2") == nullptr);
    const auto program = language_interface->parse(
        "int add(int a, int b) { return a + b; }");
    assert(language_interface->family() == ae::lang::LanguageFamily::PROGRAMMING);
    assert(program.family == ae::lang::LanguageFamily::PROGRAMMING);
    assert(program.meaning_tree.kind == ae::lang::MeaningKind::PROGRAM);
    assert(program.meaning_tree.children.size() == 1);
    const auto& function = program.meaning_tree.children.front();
    assert(function.kind == ae::lang::MeaningKind::FUNCTION);
    assert(function.text == "int add ( int a , int b )");
    assert(function.begin == 0);
    assert(function.children.back().kind == ae::lang::MeaningKind::BLOCK);
    assert(function.children.back().children.front().kind == ae::lang::MeaningKind::STATEMENT);

    AEEnginePipeline pipeline(1000, 6);
    const InferenceResult parsed_inference = pipeline.infer("replaying graph", 42, false);
    assert(parsed_inference.morphology.language == ae::lang::Language::ENGLISH);

    // Путь, который печатает AST, обязан быть настоящим путём в остовном дереве.
    const auto& tg = pipeline.thought_graph();
    assert(!parsed_inference.optimal_path.empty());
    assert(parsed_inference.optimal_path.front() == 0);
    assert(parsed_inference.optimal_path.back() ==
           AEEnginePipeline::DEFAULT_SUBGRAPH_NODES - 1);
    assert(parsed_inference.path_bottleneck > 0);
    for (std::size_t i = 1; i < parsed_inference.optimal_path.size(); ++i) {
        const int u = parsed_inference.optimal_path[i - 1];
        const int v = parsed_inference.optimal_path[i];
        assert(tg.up[v][0] == u || tg.up[u][0] == v);
        assert(tg.depth[u] != tg.depth[v]);
    }
    // Поток и цепочка должны переживать отсечение по маске.
    assert(parsed_inference.max_flow > 0);

    // max_flow агрегирует параллельные семантические цепочки: запрос с
    // кластерами обязан пропускать больше, чем пустой/неизвестный.
    const int empty_flow = pipeline.infer("", 42, false).max_flow;
    const int graph_flow = pipeline.infer("граф знаний", 42, false).max_flow;
    assert(graph_flow > empty_flow);
    assert(pipeline.infer("zzzzz", 42, false).max_flow == empty_flow);

    // Выход на естественном языке обязан пережить обратный разбор парсером.
    const auto nl_ru = pipeline.infer("нейроны графа знаний и поток данных", 42, false)
                           .natural_language;
    assert(!nl_ru.text.empty());
    assert(nl_ru.consistent);
    assert(nl_ru.cluster_recall >= 0.99);
    assert(nl_ru.reparsed_morphemes > 0);
    assert(nl_ru.reparsed.language == ae::lang::Language::RUSSIAN);
    assert((nl_ru.parsed_clusters &
            MorphemicPatriciaTrie::cluster_mask(MorphemicPatriciaTrie::Semantic::GRAPH)) != 0);
    assert((nl_ru.parsed_clusters &
            MorphemicPatriciaTrie::cluster_mask(MorphemicPatriciaTrie::Semantic::KNOW)) != 0);

    const auto nl_en = pipeline.infer("graph of knowledge", 42, false).natural_language;
    assert(!nl_en.text.empty());
    assert(nl_en.consistent);
    assert(nl_en.reparsed.language == ae::lang::Language::ENGLISH);

    // Неизвестный запрос не должен выдумывать домены.
    const auto nl_unknown = pipeline.infer("zzz qqq", 42, false).natural_language;
    assert(nl_unknown.expected_clusters == 0);
    assert(nl_unknown.parsed_clusters == 0);
    assert(nl_unknown.consistent);

    TaylorIntentKernel kernel;
    std::array<float, TaylorIntentKernel::INPUT_DIM> x{};
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = -0.5f + 0.03f * static_cast<float>(i);
    auto y_taylor = kernel.forward_taylor(x);
    auto y_std = kernel.forward_transcendental(x);
    for (std::size_t i = 0; i < y_taylor.size(); ++i) {
        assert(std::abs(y_taylor[i] - y_std[i]) < 0.01f);
    }

    MorphemicPatriciaTrie trie;
    assert(trie.morpheme_count() >= 500);
    auto [mask, count] = trie.compile_morpheme_mask("переподключение графа мыслей", y_taylor);
    assert(count >= 3);
    assert(mask != 0);
    assert((mask & MorphemicPatriciaTrie::type_mask(
                        MorphemicPatriciaTrie::MorphemeType::PREFIX)) != 0);

    const auto [rich_mask, rich_count] = trie.compile_morpheme_mask(
        "нейроны графа знаний", y_taylor);
    assert(rich_count >= 3);
    assert((rich_mask & MorphemicPatriciaTrie::cluster_mask(
                                MorphemicPatriciaTrie::Semantic::GRAPH)) != 0);
    assert((rich_mask & MorphemicPatriciaTrie::cluster_mask(
                                MorphemicPatriciaTrie::Semantic::KNOW)) != 0);
    assert((rich_mask & MorphemicPatriciaTrie::cluster_mask(
                                MorphemicPatriciaTrie::Semantic::MIND)) != 0);

    const auto [code_mask, code_count] = trie.compile_morpheme_mask(
        "компиляция и отладка кода", y_taylor);
    assert(code_count >= 3);
    assert((code_mask & MorphemicPatriciaTrie::cluster_mask(
                                MorphemicPatriciaTrie::Semantic::COMPUTE)) != 0);

    // Счётчики кластеров сохраняют поддержку темы, которую маска теряет.
    const auto rich_analysis = trie.compile_morpheme_analysis(
        "нейроны графа знаний", y_taylor);
    assert(rich_analysis.mask == rich_mask);
    assert(rich_analysis.matched_morphemes == rich_count);
    assert(rich_analysis.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::GRAPH)] >= 1);
    assert(rich_analysis.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::KNOW)] >= 1);
    assert(rich_analysis.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::MIND)] >= 1);

    // Составные термины разрешают конфликт отдельных корней.
    const auto network = trie.compile_morpheme_analysis("network", y_taylor);
    assert(network.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)] >= 1);
    assert(network.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::GRAPH)] == 0);

    const auto neural_network = trie.compile_morpheme_analysis(
        "neural network model", y_taylor);
    assert(neural_network.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::MIND)] >
           neural_network.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)]);

    const auto routing_table = trie.compile_morpheme_analysis(
        "routing table", y_taylor);
    assert(routing_table.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)] >
           routing_table.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::FLOW)]);

    const auto network_flow = trie.compile_morpheme_analysis(
        "network flow", y_taylor);
    assert(network_flow.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::FLOW)] >
           network_flow.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)]);

    const auto russian_routing = trie.compile_morpheme_analysis(
        "таблица маршрутов", y_taylor);
    assert(russian_routing.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)] >
           russian_routing.cluster_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::FLOW)]);

    // Дерево предложения: чанки, головы и поддержка по поддеревьям.
    const auto topology_tree = trie.compile_sentence_analysis(
        "network topology", y_taylor);
    assert(topology_tree.tokens.size() == 2);
    assert(topology_tree.max_depth >= 2);
    assert(topology_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::GRAPH)] >
           topology_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)]);
    assert(topology_tree.chunk_max_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::GRAPH)] >= 2);
    assert(!topology_tree.rotations.empty());
    for (const auto& node : topology_tree.nodes) {
        int current = static_cast<int>(&node - topology_tree.nodes.data());
        int guard = 0;
        while (current >= 0 && guard++ <= static_cast<int>(topology_tree.nodes.size())) {
            current = topology_tree.nodes[static_cast<std::size_t>(current)].normalized_parent;
        }
        assert(guard <= static_cast<int>(topology_tree.nodes.size()));
    }

    const auto neural_tree = trie.compile_sentence_analysis(
        "neural network", y_taylor);
    assert(neural_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::MIND)] >
           neural_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)]);

    const auto routing_tree = trie.compile_sentence_analysis(
        "routing table", y_taylor);
    assert(routing_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)] >
           routing_tree.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::FLOW)]);

    // Два чанка, соединённых союзом "and": дерево не смешивает их головы.
    const auto two_chunks = trie.compile_sentence_analysis(
        "graph of network links and routing table", y_taylor);
    assert(two_chunks.nodes.size() > 2);
    assert(two_chunks.max_depth >= 3);
    assert(two_chunks.head_counts[static_cast<std::size_t>(
               MorphemicPatriciaTrie::Semantic::NET)] >= 2);
    std::size_t chunk_nodes = 0;
    for (const auto& node : two_chunks.nodes) {
        if (node.depth == 1) ++chunk_nodes;
    }
    assert(chunk_nodes == 2);

    // Слово вне словаря не должно давать ложных морфем ("кот" -> "от").
    const auto [unknown_mask, unknown_count] =
        trie.compile_morpheme_mask("кот", y_taylor);
    assert(unknown_count == 0);
    assert((unknown_mask & 0xFFFFFF000000ULL) == 0); // нет битов типов и кластеров

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
