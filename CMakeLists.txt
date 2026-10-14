cmake_minimum_required(VERSION 3.20)
project(AEArchEngine
    VERSION 1.1.0
    DESCRIPTION "AE Arch Engine (AE::mind1_v1.1_Stable) — C++23 Cognitive Graph & AST Engine"
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AE_BUILD_SHARED "Build shared library libae_mind1.so with C ABI" ON)
option(AE_BUILD_BENCHMARK "Build CLI stress benchmark executable ae_bench" ON)
option(AE_BUILD_TESTS "Build C++23 and Python unit tests" ON)
option(AE_BUILD_PYBIND11 "Build optional native pybind11 Python extension" OFF)
option(AE_ENABLE_NATIVE_OPT "Enable -O3 -march=native CPU optimizations" ON)

find_package(Threads REQUIRED)

add_library(ae_mind1_core STATIC
    src/engine_pipeline.cpp
    src/ae_c_api.cpp
)
target_include_directories(ae_mind1_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(ae_mind1_core PUBLIC Threads::Threads)

if(AE_ENABLE_NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ae_mind1_core PUBLIC -O3 -march=native -Wall -Wextra)
endif()

if(AE_BUILD_SHARED)
    add_library(ae_mind1 SHARED
        src/engine_pipeline.cpp
        src/ae_c_api.cpp
    )
    target_include_directories(ae_mind1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(ae_mind1 PUBLIC Threads::Threads)
    if(AE_ENABLE_NATIVE_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ae_mind1 PRIVATE -O3 -march=native -Wall -Wextra)
    endif()
    set_target_properties(ae_mind1 PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        OUTPUT_NAME "ae_mind1"
    )
endif()

if(AE_BUILD_BENCHMARK)
    add_executable(ae_bench src/main_bench.cpp)
    target_link_libraries(ae_bench PRIVATE ae_mind1_core Threads::Threads)
endif()

if(AE_BUILD_PYBIND11)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(ae_mind1_pybind bindings/pybind_module.cpp)
    target_link_libraries(ae_mind1_pybind PRIVATE ae_mind1_core)
endif()

if(AE_BUILD_TESTS)
    enable_testing()
    add_executable(ae_tests tests/test_ae_core.cpp)
    target_link_libraries(ae_tests PRIVATE ae_mind1_core Threads::Threads)
    add_test(NAME AE_CppCoreTests COMMAND ae_tests)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND AE_BUILD_SHARED)
        add_test(
            NAME AE_PythonBindingsTests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py
        )
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ae_mind1_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if(AE_BUILD_SHARED)
    install(TARGETS ae_mind1 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(AE_BUILD_BENCHMARK)
    install(TARGETS ae_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ae DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
