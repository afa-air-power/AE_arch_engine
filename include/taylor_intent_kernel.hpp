#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Module 3.1: Taylor-Intent Kernel (150 neurons, 5th-order Taylor polynomials)
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

namespace ae::mind {

struct TaylorIntentKernel {
    static constexpr std::size_t INPUT_DIM = 32;
    static constexpr std::size_t HIDDEN_DIM = 150;
    static constexpr std::size_t INTENT_DIM = 16;

    alignas(64) std::array<float, INPUT_DIM * HIDDEN_DIM> W1{};
    alignas(64) std::array<float, HIDDEN_DIM> b1{};
    alignas(64) std::array<float, HIDDEN_DIM * INTENT_DIM> W2{};
    alignas(64) std::array<float, INTENT_DIM> b2{};

    explicit TaylorIntentKernel(unsigned int seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist1(0.0f, 0.15f);
        std::normal_distribution<float> dist2(0.0f, 0.10f);
        for (auto& w : W1) w = dist1(rng);
        for (auto& b : b1) b = 0.01f;
        for (auto& w : W2) w = dist2(rng);
        for (auto& b : b2) b = 0.01f;
    }

    static constexpr inline float taylor_tanh(float x) noexcept {
        if (x > 1.5f) return 0.905148f;
        if (x < -1.5f) return -0.905148f;
        const float x2 = x * x;
        return x * (1.0f - x2 * (0.33333334f - 0.13333334f * x2));
    }

    static constexpr inline float taylor_gelu(float x) noexcept {
        constexpr float SQRT_2_OVER_PI = 0.79788456f;
        constexpr float COEFF = 0.044715f;
        const float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        return 0.5f * x * (1.0f + taylor_tanh(inner));
    }

    inline std::array<float, INTENT_DIM> forward_taylor(
        const std::array<float, INPUT_DIM>& x) const noexcept {
        alignas(64) std::array<float, HIDDEN_DIM> hidden{};
        for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
            float sum = b1[i];
            const std::size_t row = i * INPUT_DIM;
            for (std::size_t j = 0; j < INPUT_DIM; ++j) {
                sum += W1[row + j] * x[j];
            }
            hidden[i] = taylor_gelu(sum);
        }

        std::array<float, INTENT_DIM> out{};
        for (std::size_t k = 0; k < INTENT_DIM; ++k) {
            float sum = b2[k];
            const std::size_t row = k * HIDDEN_DIM;
            for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
                sum += W2[row + i] * hidden[i];
            }
            out[k] = taylor_tanh(sum);
        }
        return out;
    }

    inline std::array<float, INTENT_DIM> forward_transcendental(
        const std::array<float, INPUT_DIM>& x) const noexcept {
        alignas(64) std::array<float, HIDDEN_DIM> hidden{};
        for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
            float sum = b1[i];
            const std::size_t row = i * INPUT_DIM;
            for (std::size_t j = 0; j < INPUT_DIM; ++j) {
                sum += W1[row + j] * x[j];
            }
            const float inner = 0.79788456f * (sum + 0.044715f * sum * sum * sum);
            hidden[i] = 0.5f * sum * (1.0f + std::tanh(inner));
        }

        std::array<float, INTENT_DIM> out{};
        for (std::size_t k = 0; k < INTENT_DIM; ++k) {
            float sum = b2[k];
            const std::size_t row = k * HIDDEN_DIM;
            for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
                sum += W2[row + i] * hidden[i];
            }
            out[k] = std::tanh(sum);
        }
        return out;
    }
};

} // namespace ae::mind
