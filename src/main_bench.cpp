#include "../include/engine_pipeline.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <vector>

static double get_current_rss_mb() {
    std::ifstream statm("/proc/self/statm");
    long size = 0, resident = 0;
    if (statm >> size >> resident) {
        long page_size_kb = ::sysconf(_SC_PAGESIZE) / 1024;
        return static_cast<double>(resident * page_size_kb) / 1024.0;
    }
    return 0.0;
}

int main() {
    using namespace ae::mind1;
    using namespace std::chrono;

    std::cout << "================================================================================\n";
    std::cout << " AE ARCH ENGINE (AE::mind1_v1.1_Stable) — C++23 MODULAR BUILD & BENCHMARK\n";
    std::cout << "================================================================================\n\n";

    TaylorIntentKernel intent_kernel;
    std::array<float, TaylorIntentKernel::INPUT_DIM> sample_input{};
    for (std::size_t i = 0; i < sample_input.size(); ++i) {
        sample_input[i] = -0.8f + 0.05f * static_cast<float>(i);
    }

    constexpr int KERNEL_ITERS = 100000;
    float checksum_taylor = 0.0f;
    auto t0 = high_resolution_clock::now();
    for (int it = 0; it < KERNEL_ITERS; ++it) {
        sample_input[0] = static_cast<float>(it & 15) * 0.05f;
        auto out = intent_kernel.forward_taylor(sample_input);
        checksum_taylor += out[0];
    }
    auto t1 = high_resolution_clock::now();
    double taylor_ns = duration<double, std::nano>(t1 - t0).count() / KERNEL_ITERS;

    float checksum_std = 0.0f;
    auto t2 = high_resolution_clock::now();
    for (int it = 0; it < KERNEL_ITERS; ++it) {
        sample_input[0] = static_cast<float>(it & 15) * 0.05f;
        auto out = intent_kernel.forward_transcendental(sample_input);
        checksum_std += out[0];
    }
    auto t3 = high_resolution_clock::now();
    double std_ns = duration<double, std::nano>(t3 - t2).count() / KERNEL_ITERS;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[1] Taylor-Intent Kernel (150 neurons): " << taylor_ns << " ns/call vs std::tanh "
              << std_ns << " ns/call (Speedup: " << (std_ns / std::max(1.0, taylor_ns)) << "x, RSS: "
              << get_current_rss_mb() << " MB)\n";

    AEEnginePipeline engine(50000, 6);
    std::string_view query = "переподключение графа мыслей к центроидному дереву памяти и генерация аст токенов";
    auto result = engine.infer(query, 14158, true);

    std::cout << "[2] End-to-End Pipeline Latency      : " << result.latency_us << " us\n";
    std::cout << "[3] Trace                            : " << result.ast.explainable_trace << "\n\n";
    std::cout << result.ast.ast_text << "\n";
    return (checksum_taylor == checksum_std + 1e9f) ? 1 : 0;
}
