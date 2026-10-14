#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ae/engine_pipeline.hpp"

namespace py = pybind11;
using namespace ae::mind1;

PYBIND11_MODULE(ae_mind1_pybind, m) {
    m.doc() = "AE Arch Engine (AE::mind1_v1.1_Stable) C++23 native pybind11 bindings";

    py::class_<TimeRayReport>(m, "TimeRayReport")
        .def_readonly("forward_cascade_score", &TimeRayReport::forward_cascade_score)
        .def_readonly("backward_consistency_score", &TimeRayReport::backward_consistency_score)
        .def_readonly("anomalies_pruned", &TimeRayReport::anomalies_pruned)
        .def_readonly("elapsed_us", &TimeRayReport::elapsed_us);

    py::class_<ASTPromptResult>(m, "ASTPromptResult")
        .def_readonly("ast_text", &ASTPromptResult::ast_text)
        .def_readonly("estimated_tokens", &ASTPromptResult::estimated_tokens)
        .def_readonly("explainable_trace", &ASTPromptResult::explainable_trace);

    py::class_<InferenceResult>(m, "InferenceResult")
        .def_readonly("intent_vector", &InferenceResult::intent_vector)
        .def_readonly("morpheme_mask", &InferenceResult::morpheme_mask)
        .def_readonly("matched_morphemes", &InferenceResult::matched_morphemes)
        .def_readonly("centroid_id", &InferenceResult::centroid_id)
        .def_readonly("centroid_hops", &InferenceResult::centroid_hops)
        .def_readonly("max_flow", &InferenceResult::max_flow)
        .def_readonly("path_bottleneck", &InferenceResult::path_bottleneck)
        .def_readonly("optimal_path", &InferenceResult::optimal_path)
        .def_readonly("time_ray", &InferenceResult::time_ray)
        .def_readonly("ast", &InferenceResult::ast)
        .def_readonly("latency_us", &InferenceResult::latency_us);

    py::class_<AEEnginePipeline>(m, "AEEnginePipeline")
        .def(py::init<int, int>(), py::arg("rag_nodes") = 100000, py::arg("rag_avg_degree") = 6)
        .def("reroot_thought_graph", &AEEnginePipeline::reroot_thought_graph, py::arg("new_root"))
        .def("infer", &AEEnginePipeline::infer,
             py::arg("query"),
             py::arg("target_hint") = -1,
             py::arg("run_time_ray_async") = true);
}
