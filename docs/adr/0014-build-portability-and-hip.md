# ADR-0014. Сборка и переносимость: C++23, CMake+Make, HIP/ROCm

- **Статус:** Accepted
- **Дата:** 2026-11-19
- **Модули:** `CMakeLists.txt`, `Makefile`, `training/ae_graph_hip.py`

## Контекст

Проект должен собираться:

- на обычном Linux-ноутбуке без GPU и без ROCm;
- как разделяемая библиотека с C ABI для FFI;
- как CLI-бенчмарк и набор тестов;
- с опциональным pybind11 и опциональной HIP-проверкой;
- на AMD APU Vega 7 (`gfx90c`), где нативный code object падает в runtime.

При этом пользователь не должен обязан ставить CMake, если хочет быструю
сборку, и не должен обязан ставить ROCm, если ему нужен только CPU.

## Решение

1. **C++23** (`CMAKE_CXX_STANDARD 23`, `CXX_STANDARD_REQUIRED ON`).
2. **CMake как основная система:**
   - `AE_BUILD_SHARED` (по умолчанию `ON`) — `libae_mind1.so` с C ABI;
   - `AE_BUILD_BENCHMARK` (`ON`) — `ae_bench`;
   - `AE_BUILD_TESTS` (`ON`) — C++/Python тесты;
   - `AE_BUILD_PYBIND11` (`OFF`) — опциональный нативный модуль;
   - `AE_ENABLE_NATIVE_OPT` (`ON`) — `-O3 -march=native`;
   - `AE_ENABLE_HIP_TESTS` (`OFF`) — регистрация HIP smoke test в CTest.
3. **Автономный `Makefile`** — быстрая сборка без CMake (`make all`,
   `make test`, `make bench`).
4. **HIP/ROCm — опция, а не зависимость.** `training/ae_graph_hip.py`
   компилирует kernel через `hipcc --offload-arch=<target>`, проверяет
   выделение памяти, H2D, dispatch, D2H и корректность данных. По умолчанию
   применяется `HSA_OVERRIDE_GFX_VERSION=9.0.0` и `--target gfx900`.
5. **Отключаемость.** `--hsa-override ""` позволяет проверить нативный
   target; `AE_ENABLE_HIP_TESTS=OFF` полностью убирает HIP из CTest.

## Обоснование

- **Почему C++23.** Нужны `std::popcount`, `std::bit`, `starts_with`,
  `std::span`-подобные идиомы и улучшенный `constexpr`. Это упрощает
  компактный и переносимый код без внешних библиотек.
- **Почему CMake + Make.** CMake даёт нормальную интеграцию с IDE, CTest,
  pybind11 и опциями; Makefile — минимальный путь для быстрой сборки и
  окружений без CMake. Дублирование небольшое, зато покрывает оба сценария.
- **Почему `libae_mind1.so` и C ABI.** Одна разделяемая библиотека — это
  единица поставки для Python/Rust/Go (см. [ADR-0011](0011-c-abi-and-python-bindings.md)).
- **Почему `-march=native` по умолчанию.** Ядро ориентировано на локальный
  стенд и бенчмарки; флаг даёт заметный выигрыш. Для переносимых сборок его
  можно отключить (`AE_ENABLE_NATIVE_OPT=OFF`).
- **Почему HIP опционален.** ROCm тяжёл, нестабилен на APU и не нужен для
  CPU-контура. Smoke test проверяет устройство отдельно и не блокирует
  основную сборку.
- **Почему override `9.0.0` на Vega 7.** На тестируемом APU (`gfx90c`,
  XNACK enabled, ROCm 7.16) нативный code object аварийно завершается в
  runtime. Классический override на `gfx900` — рабочая конфигурация; скрипт
  применяет её по умолчанию, но оставляет возможность отключить.

## Последствия

- **Выигрыш:** проект собирается и тестируется без GPU; HIP проверяется
  отдельно; FFI-библиотека собирается одной командой; есть fallback без
  CMake.
- **Цена:** `-march=native` делает бинарник непереносимым между CPU;
  для дистрибуции нужна сборка с `AE_ENABLE_NATIVE_OPT=OFF` или явным
  `-march`.
- **Ограничение:** HIP-backend обучения не реализован; smoke test
  подтверждает только работоспособность устройства и компилятора.
- **Ограничение:** `mmap` в Mega RAG — POSIX-зависимость; Windows-порт
  потребует замены на `VirtualAlloc`/`CreateFileMapping`.
- **Ограничение:** `-march=native` и `-O3` могут менять порядок
  floating-point операций между платформами; для строгой воспроизводимости
  результатов нужна фиксированная сборка.

## Ссылки

- CMake: `CMakeLists.txt`.
- Make: `Makefile`.
- HIP: `training/ae_graph_hip.py`.
- Документация: `docs/README.md`, `docs/TRAINING.md` (раздел «Vega 7 / HIP»).
- CI-команда:
  `cmake -S . -B build -DAE_BUILD_PYBIND11=ON -DAE_ENABLE_HIP_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
  → `cmake --build build -j` → `ctest --test-dir build --output-on-failure`.
