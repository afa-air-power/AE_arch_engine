# AE Arch Engine (`AE::mind1_v1.1_Stable`)

Модульный C++23 проект вычислительного ядра **AE Arch Engine**, система сборки **CMake / Make**, **C ABI** (`libae_mind1.so`) и **Python-обвязка** (`ctypes` + `pybind11` + оркестратор локальной SLM 1.5B–3B).

Автор концепции и архитектуры: **Фирстов Андрей Александрович (ИУ5 МГТУ им. Н.Э. Баумана)**.

## Структура проекта

```text
AE_Arch_Engine/
├── CMakeLists.txt                  # Конфигурация сборки CMake (C++23, libae_mind1.so, ae_bench, ae_tests, pybind11)
├── Makefile                        # Автономный Makefile для быстрой сборки без CMake
├── pyproject.toml                  # Метаданные Python-пакета обвязки
├── README.md                       # Документация по сборке и API
├── include/
│   ├── ae/lang/                    # Общий интерфейс и типы языковых деревьев
│   │   ├── parser.hpp              # LanguageInterface, ParseResult, MeaningNode
│   │   ├── interface.hpp           # Общие адаптеры human/sience/verbal/programing
│   │   ├── en/parser.hpp           # API английского морфемного парсера
│   │   ├── ru/parser.hpp           # API русского морфемного парсера
│   │   └── programing/parser.hpp   # API дерева разбора исходного кода
│   ├── taylor_intent_kernel.hpp    # Модуль 3.1: Taylor-Intent Kernel (150 нейронов, полиномы Тейлора 5-го порядка)
│   ├── morphemic_trie.hpp          # Модуль 3.2: Morphemic Patricia Trie и 64-битная булева маска сочетаемости
│   ├── mega_rag_centroid.hpp       # Модуль 3.3: Mega RAG (CSR в mmap + Центроидная декомпозиция за O(log N))
│   ├── thought_graph.hpp           # Модуль 4: Топология Графа Мыслей (DSU Краскал, BINUP LCA, динамический reroot)
│   ├── dinitz_decoder.hpp          # Модуль 3.4: Zero-Allocation кэш-локальный максимальный поток Диница
│   ├── time_ray_ast.hpp            # Модули 3.5 и 6: Async Time-Ray Smoothing и компилятор верифицированного AST (~400 токенов)
│   ├── engine_pipeline.hpp         # Единый высокоуровневый конвейер AEEnginePipeline
│   └── ae_c_api.h                  # Чистый C ABI интерфейс (extern "C") для FFI (Python ctypes, Rust, Go)
├── src/
│   ├── lang_parser.cpp            # Определение языка и маршрутизация к парсеру
│   ├── engine_pipeline.cpp         # Реализация сквозного конвейера AEEnginePipeline
│   ├── ae_c_api.cpp                # Экспортируемые функции разделяемой библиотеки libae_mind1.so
│   └── main_bench.cpp              # CLI-бенчмарк производительности (ae_bench)
├── lang/
│   ├── humans/                    # Разбор русских и английских морфем
│   └── programing/parser.cpp      # Лексико-синтаксическое дерево программного кода
├── bindings/
│   ├── ae_engine.py                # Python SDK (AEArchEngine) поверх libae_mind1.so (без внешних зависимостей)
│   ├── slm_bridge.py               # Оркестратор верифицированного AST-промпта для локальной SLM 1.5B–3B
│   └── pybind_module.cpp           # Опциональный нативный модуль pybind11 (ae_mind1_pybind)
└── tests/
    ├── test_ae_core.cpp            # Модульные тесты C++23 ядра и C ABI
    └── test_bindings.py            # Интеграционные тесты Python-обвязки и SLM-моста
```

## Сборка через CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAE_ENABLE_NATIVE_OPT=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Для включения нативного модуля `pybind11`:

```bash
cmake -B build -DAE_BUILD_PYBIND11=ON
cmake --build build -j$(nproc)
```

## Быстрая сборка через Make (без CMake)

```bash
make all
make test
make bench
```

## Общий интерфейс языков

`ae::lang::LanguageInterface` — единый интерфейс `parse`, `language` и `family`;
результат содержит морфемы (для human-языков) и общий `MeaningNode` для
древовидного представления. Один интерфейс доступен из пространств имён
`ae::lang::human`, `ae::lang::sience`, `ae::lang::verbal` и
`ae::lang::programing`. Программный парсер строит дерево программы, функций,
блоков, объявлений, управляющих конструкций и лексических узлов.

`ae::lang::select_parser(family, text)` выбирает парсер по семейству; для
`SCIENCE` и `VERBAL` возвращается `nullptr`, пока эти семейства не получат
реализацию. Одноаргументный `select_parser(text)` остаётся совместимым
маршрутизатором русского/английского human-парсера.
