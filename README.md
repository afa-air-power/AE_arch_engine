# AE Arch Engine (`AE::mind_v1.1_Stable`)

Модульный C++23 проект вычислительного ядра: CMake/Make, C ABI (`libae_mind1.so`),
Python-обвязка (`ctypes` + `pybind11`) и обучаемый графовый контур поверх
алгоритмов ядра.

Автор концепции и архитектуры: **Фирстов Андрей Александрович (ИУ5 МГТУ им. Н.Э. Баумана)**.

## Документация

- [docs/README.md](docs/README.md) — сборка, структура проекта, API и Python-обвязка.
- [docs/TRAINING.md](docs/TRAINING.md) — признаки из алгоритмов, датасет, обучение,
  предметные метрики и HIP-проверка Vega 7.

## Быстрый старт

```bash
cmake -S . -B build -DAE_BUILD_PYBIND11=ON -DAE_ENABLE_HIP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Vega 7 (`gfx90c`, XNACK enabled):

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 -m training.ae_graph_hip --target gfx900
```
