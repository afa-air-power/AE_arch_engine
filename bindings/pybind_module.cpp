#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/engine_pipeline.hpp"

namespace py = pybind11;
using namespace ae::mind;
namespace lang = ae::lang;

PYBIND11_MODULE(ae_mind_pybind, m) {
    m.doc() = "AE Arch Engine (AE::mind_v1.1_Stable) C++23 native pybind11 bindings";

    py::class_<TimeRayReport>(m, "TimeRayReport")
        .def_readonly("forward_cascade_score", &TimeRayReport::forward_cascade_score)
        .def_readonly("backward_consistency_score", &TimeRayReport::backward_consistency_score)
        .def_readonly("anomalies_pruned", &TimeRayReport::anomalies_pruned)
        .def_readonly("elapsed_us", &TimeRayReport::elapsed_us);

    py::class_<ASTPromptResult>(m, "ASTPromptResult")
        .def_readonly("ast_text", &ASTPromptResult::ast_text)
        .def_readonly("estimated_tokens", &ASTPromptResult::estimated_tokens)
        .def_readonly("explainable_trace", &ASTPromptResult::explainable_trace);

    py::class_<MorphemicPatriciaTrie::SentenceToken>(m, "SentenceToken")
        .def_readonly("text", &MorphemicPatriciaTrie::SentenceToken::text)
        .def_readonly("begin", &MorphemicPatriciaTrie::SentenceToken::begin)
        .def_readonly("end", &MorphemicPatriciaTrie::SentenceToken::end)
        .def_readonly("function_word", &MorphemicPatriciaTrie::SentenceToken::function_word)
        .def_readonly("starts_chunk", &MorphemicPatriciaTrie::SentenceToken::starts_chunk)
        .def_property_readonly("cluster_counts", [](const MorphemicPatriciaTrie::SentenceToken& token) {
            return std::vector<std::uint16_t>(token.analysis.cluster_counts.begin(),
                                               token.analysis.cluster_counts.end());
        });

    py::class_<MorphemicPatriciaTrie::SentenceNode>(m, "SentenceNode")
        .def_readonly("head_token", &MorphemicPatriciaTrie::SentenceNode::head_token)
        .def_readonly("parent", &MorphemicPatriciaTrie::SentenceNode::parent)
        .def_readonly("normalized_parent", &MorphemicPatriciaTrie::SentenceNode::normalized_parent)
        .def_readonly("depth", &MorphemicPatriciaTrie::SentenceNode::depth)
        .def_readonly("normalized_depth", &MorphemicPatriciaTrie::SentenceNode::normalized_depth)
        .def_readonly("begin_token", &MorphemicPatriciaTrie::SentenceNode::begin_token)
        .def_readonly("end_token", &MorphemicPatriciaTrie::SentenceNode::end_token)
        .def_property_readonly("cluster_counts", [](const MorphemicPatriciaTrie::SentenceNode& node) {
            return std::vector<std::uint16_t>(node.counts.begin(), node.counts.end());
        });

    py::class_<MorphemicPatriciaTrie::TreeRotation>(m, "TreeRotation")
        .def_readonly("child", &MorphemicPatriciaTrie::TreeRotation::child)
        .def_readonly("parent", &MorphemicPatriciaTrie::TreeRotation::parent)
        .def_readonly("pivot", &MorphemicPatriciaTrie::TreeRotation::pivot)
        .def_readonly("direction", &MorphemicPatriciaTrie::TreeRotation::direction)
        .def_readonly("before_depth", &MorphemicPatriciaTrie::TreeRotation::before_depth)
        .def_readonly("after_depth", &MorphemicPatriciaTrie::TreeRotation::after_depth);

    py::class_<MorphemicPatriciaTrie::SentenceAnalysis>(m, "SentenceAnalysis")
        .def_readonly("tokens", &MorphemicPatriciaTrie::SentenceAnalysis::tokens)
        .def_readonly("nodes", &MorphemicPatriciaTrie::SentenceAnalysis::nodes)
        .def_readonly("rotations", &MorphemicPatriciaTrie::SentenceAnalysis::rotations)
        .def_readonly("normalized_root", &MorphemicPatriciaTrie::SentenceAnalysis::normalized_root)
        .def_property_readonly("root_counts", [](const MorphemicPatriciaTrie::SentenceAnalysis& analysis) {
            return std::vector<std::uint16_t>(analysis.root_counts.begin(),
                                               analysis.root_counts.end());
        })
        .def_property_readonly("head_counts", [](const MorphemicPatriciaTrie::SentenceAnalysis& analysis) {
            return std::vector<std::uint16_t>(analysis.head_counts.begin(),
                                               analysis.head_counts.end());
        })
        .def_property_readonly("chunk_max_counts", [](const MorphemicPatriciaTrie::SentenceAnalysis& analysis) {
            return std::vector<std::uint16_t>(analysis.chunk_max_counts.begin(),
                                               analysis.chunk_max_counts.end());
        })
        .def_readonly("max_depth", &MorphemicPatriciaTrie::SentenceAnalysis::max_depth);

    py::class_<NaturalLanguageReport>(m, "NaturalLanguageReport")
        .def_readonly("text", &NaturalLanguageReport::text)
        .def_readonly("feedback", &NaturalLanguageReport::feedback)
        .def_readonly("cluster_recall", &NaturalLanguageReport::cluster_recall)
        .def_readonly("cluster_precision", &NaturalLanguageReport::cluster_precision)
        .def_readonly("reparsed_morphemes", &NaturalLanguageReport::reparsed_morphemes)
        .def_readonly("refinement_passes", &NaturalLanguageReport::refinement_passes)
        .def_readonly("consistent", &NaturalLanguageReport::consistent);

    py::class_<InferenceResult>(m, "InferenceResult")
        .def_readonly("intent_vector", &InferenceResult::intent_vector)
        .def_readonly("sentence", &InferenceResult::sentence)
        .def_readonly("morpheme_mask", &InferenceResult::morpheme_mask)
        .def_readonly("matched_morphemes", &InferenceResult::matched_morphemes)
        .def_readonly("centroid_id", &InferenceResult::centroid_id)
        .def_readonly("centroid_hops", &InferenceResult::centroid_hops)
        .def_readonly("max_flow", &InferenceResult::max_flow)
        .def_readonly("path_bottleneck", &InferenceResult::path_bottleneck)
        .def_readonly("optimal_path", &InferenceResult::optimal_path)
        .def_readonly("time_ray", &InferenceResult::time_ray)
        .def_readonly("ast", &InferenceResult::ast)
        .def_readonly("natural_language", &InferenceResult::natural_language)
        .def_readonly("latency_us", &InferenceResult::latency_us);

    // ---- Algorithm-derived feature vector -------------------------------
    py::class_<AlgorithmFeatureVector>(m, "AlgorithmFeatureVector")
        .def_readonly("values", &AlgorithmFeatureVector::values)
        .def_static("dim", []() { return AlgorithmFeatureVector::DIM; })
        .def_static("names", []() {
            std::vector<std::string> out;
            out.reserve(kAlgorithmFeatureNames.size());
            for (std::string_view name : kAlgorithmFeatureNames) {
                out.emplace_back(name);
            }
            return out;
        })
        .def("__len__", [](const AlgorithmFeatureVector&) {
            return AlgorithmFeatureVector::DIM;
        })
        .def("__getitem__", [](const AlgorithmFeatureVector& self, int index) {
            if (index < 0 || index >= AlgorithmFeatureVector::DIM) {
                throw py::index_error("feature index out of range");
            }
            return self[index];
        });

    // ---- Thought graph topology -----------------------------------------
    py::class_<ThoughtGraphTopology::Edge>(m, "ThoughtEdge")
        .def_readonly("u", &ThoughtGraphTopology::Edge::u)
        .def_readonly("v", &ThoughtGraphTopology::Edge::v)
        .def_readonly("weight", &ThoughtGraphTopology::Edge::weight)
        .def_readonly("id", &ThoughtGraphTopology::Edge::id)
        .def_readonly("tag", &ThoughtGraphTopology::Edge::tag);

    py::class_<ThoughtGraphTopology>(m, "ThoughtGraphTopology")
        .def_readonly("n", &ThoughtGraphTopology::n)
        .def_readonly("log_n", &ThoughtGraphTopology::log_n)
        .def_readonly("current_root", &ThoughtGraphTopology::current_root)
        .def_readonly("depth", &ThoughtGraphTopology::depth)
        .def_readonly("all_edges", &ThoughtGraphTopology::all_edges)
        .def_readonly("adj_mst", &ThoughtGraphTopology::adj_mst)
        .def_readonly("adj_horiz", &ThoughtGraphTopology::adj_horiz)
        .def("reroot", &ThoughtGraphTopology::reroot)
        .def("query_path_binup", &ThoughtGraphTopology::query_path_binup)
        .def("query_path_min_binup", &ThoughtGraphTopology::query_path_min_binup)
        .def("path_between", &ThoughtGraphTopology::path_between);

    // ---- Morphemic trie --------------------------------------------------
    py::class_<MorphemicPatriciaTrie>(m, "MorphemicPatriciaTrie")
        .def("morpheme_count", &MorphemicPatriciaTrie::morpheme_count)
        .def("contains", &MorphemicPatriciaTrie::contains)
        .def("compile_morpheme_mask",
             [](const MorphemicPatriciaTrie& self, std::string_view text,
                const std::array<float, 16>& intent) {
                 return self.compile_morpheme_mask(text, intent);
             },
             py::arg("text"), py::arg("intent"))
        .def("compile_morpheme_analysis",
             [](const MorphemicPatriciaTrie& self, std::string_view text,
                const std::array<float, 16>& intent) {
                 const auto analysis = self.compile_morpheme_analysis(text, intent);
                 return py::make_tuple(
                     analysis.mask,
                     analysis.matched_morphemes,
                     std::vector<std::uint16_t>(analysis.cluster_counts.begin(),
                                                analysis.cluster_counts.end()));
             },
             py::arg("text"), py::arg("intent"))
        .def("compile_sentence_analysis",
             [](const MorphemicPatriciaTrie& self, std::string_view text,
                const std::array<float, 16>& intent) {
                 const auto analysis = self.compile_sentence_analysis(text, intent);
                 py::list tokens;
                 for (const auto& token : analysis.tokens) {
                     py::dict item;
                     item["text"] = std::string(token.text);
                     item["begin"] = token.begin;
                     item["end"] = token.end;
                     item["function_word"] = token.function_word;
                     item["starts_chunk"] = token.starts_chunk;
                     item["counts"] = std::vector<std::uint16_t>(
                         token.analysis.cluster_counts.begin(),
                         token.analysis.cluster_counts.end());
                     tokens.append(item);
                 }
                 py::list nodes;
                 for (const auto& node : analysis.nodes) {
                     py::dict item;
                     item["head_token"] = node.head_token;
                     item["parent"] = node.parent;
                     item["normalized_parent"] = node.normalized_parent;
                     item["depth"] = node.depth;
                     item["normalized_depth"] = node.normalized_depth;
                     item["begin_token"] = node.begin_token;
                     item["end_token"] = node.end_token;
                     item["counts"] = std::vector<std::uint16_t>(
                         node.counts.begin(), node.counts.end());
                     nodes.append(item);
                 }
                 py::dict out;
                 py::list rotations;
                 for (const auto& rotation : analysis.rotations) {
                     py::dict item;
                     item["child"] = rotation.child;
                     item["parent"] = rotation.parent;
                     item["pivot"] = rotation.pivot;
                     item["direction"] = rotation.direction;
                     item["before_depth"] = rotation.before_depth;
                     item["after_depth"] = rotation.after_depth;
                     rotations.append(item);
                 }
                 out["tokens"] = tokens;
                 out["nodes"] = nodes;
                 out["rotations"] = rotations;
                 out["normalized_root"] = analysis.normalized_root;
                 out["root_counts"] = std::vector<std::uint16_t>(
                     analysis.root_counts.begin(), analysis.root_counts.end());
                 out["head_counts"] = std::vector<std::uint16_t>(
                     analysis.head_counts.begin(), analysis.head_counts.end());
                 out["chunk_max_counts"] = std::vector<std::uint16_t>(
                     analysis.chunk_max_counts.begin(),
                     analysis.chunk_max_counts.end());
                 out["max_depth"] = analysis.max_depth;
                 return out;
             },
             py::arg("text"), py::arg("intent"));

    // ---- Mega RAG centroid graph ----------------------------------------
    py::class_<MegaRAGCentroidGraph>(m, "MegaRAGCentroidGraph")
        .def_readonly("n_nodes", &MegaRAGCentroidGraph::n_nodes)
        .def_readonly("centroid_depth", &MegaRAGCentroidGraph::centroid_depth)
        .def_readonly("centroid_parent", &MegaRAGCentroidGraph::centroid_parent)
        .def_readonly("root_centroid", &MegaRAGCentroidGraph::root_centroid)
        .def("build_centroid_index", &MegaRAGCentroidGraph::build_centroid_index)
        .def("locate_centroid_cluster", &MegaRAGCentroidGraph::locate_centroid_cluster);

    // ---- Dinitz decoder --------------------------------------------------
    py::class_<DinitzGraphDecoder>(m, "DinitzGraphDecoder")
        .def(py::init<int>(), py::arg("nodes") = 64)
        .def_readonly("n", &DinitzGraphDecoder::n)
        .def_readonly("edge_count", &DinitzGraphDecoder::edge_count)
        .def("add_static_edge", &DinitzGraphDecoder::add_static_edge,
             py::arg("from"), py::arg("to"), py::arg("base_cap"), py::arg("tag"))
        .def("apply_boolean_mask_and_reset",
             &DinitzGraphDecoder::apply_boolean_mask_and_reset, py::arg("boolean_mask"))
        .def("max_flow", &DinitzGraphDecoder::max_flow, py::arg("s"), py::arg("t"));

    // ---- Language parsers -------------------------------------------------
    m.def("detect_language", [](std::string_view text) {
        switch (lang::detect_language(text)) {
            case lang::Language::RUSSIAN: return "russian";
            case lang::Language::ENGLISH: return "english";
            default: return "programming";
        }
    }, py::arg("text"));

    m.def("parse_morphemes", [](std::string_view text) {
        std::vector<std::string> out;
        if (lang::LanguageInterface* parser = lang::select_parser(text)) {
            for (const auto& morpheme : parser->parse(text).morphemes) {
                out.push_back(morpheme.text);
            }
        }
        return out;
    }, py::arg("text"));

    // ---- Pipeline --------------------------------------------------------
    py::class_<AEEnginePipeline>(m, "AEEnginePipeline")
        .def(py::init<int, int>(), py::arg("rag_nodes") = 100000, py::arg("rag_avg_degree") = 6)
        .def("reroot_thought_graph", &AEEnginePipeline::reroot_thought_graph, py::arg("new_root"))
        .def("extract_features", &AEEnginePipeline::extract_features,
             py::arg("query"), py::arg("target_hint") = -1)
        .def("infer", &AEEnginePipeline::infer,
             py::arg("query"),
             py::arg("target_hint") = -1,
             py::arg("run_time_ray_async") = true)
        .def("thought_graph", &AEEnginePipeline::thought_graph,
             py::return_value_policy::reference_internal)
        .def("trie", &AEEnginePipeline::trie,
             py::return_value_policy::reference_internal)
        .def("rag", &AEEnginePipeline::rag,
             py::return_value_policy::reference_internal)
        .def("decoder", py::overload_cast<>(&AEEnginePipeline::decoder),
             py::return_value_policy::reference_internal);
}
