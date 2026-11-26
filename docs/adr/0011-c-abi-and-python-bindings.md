# ADR-0011. C ABI и Python-обвязка без обязательных зависимостей

- **Статус:** Accepted
- **Дата:** 2026-11-19
- **Модуль:** [include/ae_c_api.h](../../include/ae_c_api.h), [src/ae_c_api.cpp](../../src/ae_c_api.cpp), [bindings/](../../bindings/)

## Контекст

Ядро написано на C++23, но использоваться должно из Python, Rust, Go и
других языков. Требования:

- стабильная граница, не зависящая от C++ ABI и версии компилятора;
- отсутствие обязательных внешних зависимостей у Python-клиента;
- доступ к результатам, которые нельзя упаковать в плоскую структуру
  (AST-текст, trace, NL-ответ);
- безопасность владения памятью на границе;
- опциональный нативный модуль для тех, кому нужна максимальная скорость.

## Решение

1. **Чистый C ABI** (`extern "C"`):
   - `AEEngineHandle*` — непрозрачный указатель на `AEEnginePipeline`;
   - `AEInferenceMetrics` — POD-структура с фиксированными полями
     (`uint64_t`, `int32_t`, `double`, `size_t`, `path_nodes[32]`);
   - `ae_engine_create/destroy`, `ae_engine_add_morpheme`,
     `ae_engine_reroot`, `ae_engine_infer`;
   - геттеры `ae_engine_get_last_ast/trace/nl/nl_feedback` возвращают
     `const char*`, принадлежащие handle.
2. **Python SDK** (`bindings/ae_engine.py`) — `ctypes` без внешних
   зависимостей; `AEArchEngine` загружает `libae_mind1.so`, описывает
   `AEInferenceMetrics`, возвращает `AEInferenceOutput`.
3. **Опциональный pybind11-модуль** (`bindings/pybind_module.cpp`) —
   включается флагом `-DAE_BUILD_PYBIND11=ON`; нужен для нативной скорости,
   но не обязателен.
4. **SLM-мост** (`bindings/slm_bridge.py`) — оркестратор, который использует
   `nl_text` как основной ответ и `trace_response` как детальный отчёт.

## Обоснование

- **Почему C ABI, а не C++ API.** C ABI:
  - не зависит от name mangling, стандартной библиотеки и исключений;
  - одинаково доступен из `ctypes`, `cgo`, JNI, Rust `extern "C"`;
  - позволяет менять C++-реализацию, не ломая клиентов.
- **Почему `ctypes` по умолчанию.** Python-клиент не должен требовать
  `pybind11`, компилятора и заголовков. `ctypes` входит в стандартную
  библиотеку, а `libae_mind1.so` собирается один раз. pybind11 остаётся
  опцией для производительных интеграций.
- **Почему POD-структура метрик.** Плоская структура фиксированного размера:
  - копируется одним `memcpy` через FFI;
  - не содержит указателей и C++-типов;
  - версионируется полями, а не ABI.
  `path_nodes[32]` ограничивает путь; для 64-узлового подграфа этого
  достаточно, а переполнение обрабатывается усечением.
- **Почему строки — геттеры, а не поля структуры.** AST, trace и NL-ответ
  имеют переменную длину. Возврат `const char*` на внутренний буфер handle:
  - не требует выделения памяти на границе;
  - живёт до следующего `infer`/`destroy`;
  - не переусложняет структуру метрик.
  Владение явно задокументировано: клиент не освобождает строку.
- **Почему `nl_*` поля продублированы в метриках.** Клиенту нужны
  recall/precision/consistent без парсинга строки feedback; это дешёвые
  скалярные значения.

## Последствия

- **Выигрыш:** одна разделяемая библиотека для всех языков; Python-клиент
  без зависимостей; возможность включить pybind11 только там, где нужно.
- **Цена:** C ABI нужно сопровождать вручную при изменении структуры;
  комментарий в `ae_c_api.h` требует синхронизации с `bindings/ae_engine.py`.
- **Ограничение:** `const char*` действителен только до следующего вызова
  `infer` или `destroy`; клиент должен скопировать строку, если хранит её.
- **Ограничение:** `path_nodes[32]` усекает путь; для более длинных путей
  нужен новый API или увеличение константы (это изменение ABI).

## Ссылки

Точные места в коде ищутся grep-скриптом по маркерам `ADR-0011` и
`docs/adr/0011-…`; номера строк в документации не фиксируются и не
устаревают при сдвиге кода:

```bash
docs/adr/find.sh 0011
```

**Заголовок:** [include/ae_c_api.h](../../include/ae_c_api.h)
— `AEInferenceMetrics` (POD), `ae_engine_create`/`ae_engine_destroy`,
`ae_engine_add_morpheme`/`ae_engine_reroot`, `ae_engine_infer`, геттеры AST/trace/NL.

**Реализация:** [src/ae_c_api.cpp](../../src/ae_c_api.cpp)
— `create`, `destroy`, `add_morpheme`, `reroot`, `infer`, геттеры.

**Python:** [bindings/ae_engine.py](../../bindings/ae_engine.py)
(`AEArchEngine`, `infer`), [bindings/slm_bridge.py](../../bindings/slm_bridge.py),
[bindings/pybind_module.cpp](../../bindings/pybind_module.cpp).

- Сборка: [CMakeLists.txt](../../CMakeLists.txt) (`libae_mind1`), [Makefile](../../Makefile).
- Тесты: [tests/test_bindings.py](../../tests/test_bindings.py).
