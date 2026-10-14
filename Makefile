# ============================================================================
# AE Arch Engine (AE::mind1_v1.1_Stable) — Direct Makefile (Fallback for CMake)
# ============================================================================

CXX      ?= g++
CXXFLAGS ?= -std=c++23 -O3 -march=native -Wall -Wextra -fPIC -pthread -Iinclude
BUILD_DIR = build

SHARED_LIB = $(BUILD_DIR)/libae_mind1.so
BENCH_BIN  = $(BUILD_DIR)/ae_bench
TEST_BIN   = $(BUILD_DIR)/ae_tests

SRCS_LIB   = src/engine_pipeline.cpp src/ae_c_api.cpp

.PHONY: all clean test bench

all: $(SHARED_LIB) $(BENCH_BIN) $(TEST_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SHARED_LIB): $(SRCS_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared $(SRCS_LIB) -o $(SHARED_LIB)

$(BENCH_BIN): src/main_bench.cpp $(SRCS_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/main_bench.cpp $(SRCS_LIB) -o $(BENCH_BIN)

$(TEST_BIN): tests/test_ae_core.cpp $(SRCS_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_ae_core.cpp $(SRCS_LIB) -o $(TEST_BIN)

test: all
	./$(TEST_BIN)
	python3 tests/test_bindings.py

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
