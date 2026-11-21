# AE Arch Engine (`AE::mind_v1.1_Stable`)

Модульный C++23 проект вычислительного ядра **AE Arch Engine**, система сборки **CMake / Make**, **C ABI** (`libae_mind1.so`) и **Python-обвязка** (`ctypes` + `pybind11` + оркестратор локальной SLM 1.5B–3B).

Автор концепции и архитектуры: **Фирстов Андрей Александрович (ИУ5 МГТУ им. Н.Э. Баумана)**.

## Структура проекта

```text
AE_Arch_Engine/
├── CMakeLists.txt                  # Конфигурация сборки CMake (C++23, libae_mind1.so, ae_bench, ae_tests, pybind11)
├── Makefile                        # Автономный Makefile для быстрой сборки без CMake
├── pyproject.toml                  # Метаданные Python-пакета обвязки
├── README.md                       # Краткий указатель на документацию
├── docs/                           # Документация
│   ├── README.md                   # Документация по сборке и API
│   ├── TRAINING.md                 # Обучаемый контур и предметные метрики
│   ├── LABELING.md                 # Правила и цикл ручной разметки корпуса
│   ├── ACCURACY.md                 # Измеренные варианты повышения точности
│   └── adr/                        # Architecture Decision Records (обоснование решений)
├── include/                        # Публичные заголовки (устанавливаются в <prefix>/include)
│   ├── ae_c_api.h                  # Чистый C ABI (extern "C") для FFI (Python ctypes, Rust, Go)
│   ├── engine_pipeline.hpp         # Единый высокоуровневый конвейер AEEnginePipeline
│   ├── taylor_intent_kernel.hpp    # Модуль 3.1: Taylor-Intent Kernel (150 нейронов, полиномы Тейлора 5-го порядка)
│   ├── morphemic_trie.hpp          # Модуль 3.2: Morphemic Patricia Trie и 64-битная булева маска сочетаемости
│   ├── mega_rag_centroid.hpp       # Модуль 3.3: Mega RAG (CSR в mmap + Центроидная декомпозиция)
│   ├── thought_graph.hpp           # Модуль 4: Топология Графа Мыслей (DSU Краскал, BINUP LCA, динамический reroot)
│   ├── dinitz_decoder.hpp          # Модуль 3.4: Zero-Allocation кэш-локальный максимальный поток Диница
│   ├── time_ray_ast.hpp            # Модули 3.5 и 6: Async Time-Ray Smoothing и компилятор верифицированного AST
│   ├── natural_language_output.hpp # Модуль 6.1: вербализация на естественном языке + обратный разбор парсером
│   └── lang/                       # Общий интерфейс и типы языковых деревьев
│       ├── parser.hpp              # LanguageInterface, ParseResult, MeaningNode, detect/select_parser
│       ├── interface.hpp           # Алиасы human / science / verbal / programing
│       ├── detail/parser_detail.hpp# Общий алгоритм разбора слов на морфемы
│       ├── en/parser.hpp           # API английского морфемного парсера
│       ├── ru/parser.hpp           # API русского морфемного парсера
│       └── programing/parser.hpp   # API дерева разбора исходного кода
├── src/                            # Реализации (не устанавливаются)
│   ├── ae_c_api.cpp                # Экспортируемые функции разделяемой библиотеки libae_mind1.so
│   ├── engine_pipeline.cpp         # Реализация сквозного конвейера AEEnginePipeline
│   ├── lang_parser.cpp             # Определение языка и маршрутизация к парсеру
│   ├── main_bench.cpp              # CLI-бенчмарк производительности (ae_bench)
│   └── lang/
│       ├── en/parser.cpp           # Разбор английских морфем
│       ├── ru/parser.cpp           # Разбор русских морфем
│       └── programing/parser.cpp   # Лексико-синтаксическое дерево программного кода
├── bindings/
│   ├── ae_engine.py                # Python SDK (AEArchEngine) поверх libae_mind1.so (без внешних зависимостей)
│   ├── slm_bridge.py               # Оркестратор верифицированного AST-промпта для локальной SLM 1.5B–3B
│   └── pybind_module.cpp           # Опциональный нативный модуль pybind11 (ae_mind_pybind)
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

## Бенчмарки

Замеры сняты с `build/ae_bench` (цель `ae_bench`, медиана трёх прогонов).
Команда воспроизведения:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAE_ENABLE_NATIVE_OPT=ON
cmake --build build -j$(nproc)
./build/ae_bench
```

Стенд: **AMD Ryzen 5 5500U** (6 ядер / 12 потоков), **GCC 16.2.1**,
Linux 7.3.4 x86_64, сборка `Release` с `-O3 -march=native`, один поток,
горячий кэш. Абсолютные значения зависят от частоты CPU и фоновой нагрузки,
поэтому таблица годится для регрессий на этой же машине, а не для сравнения
с другими платформами.

| Блок | Операция | p50 | Пропускная способность |
|---|---|---:|---:|
| Taylor-Intent Kernel | `forward_taylor` (150 нейронов, 32→150→16) | 3.85 µs | ~260 тыс. вызовов/с |
| Taylor-Intent Kernel | `forward_transcendental` (`std::tanh`) | 6.80 µs | ~147 тыс. вызовов/с |
| Taylor-Intent Kernel | ускорение полиномиального GELU/tanh | **1.77×** | — |
| Morphemic Trie | `compile_morpheme_mask` (1804 морфемы, 2580 узлов) | 330 ns | ~3.0 млн запросов/с |
| `ru::Parser` | разбор 22 слов | 9.45 µs | ~106 тыс. разборов/с |
| `en::Parser` | разбор 7 слов | 1.14 µs | ~870 тыс. разборов/с |
| `programing::Parser` | разбор C-подобного фрагмента | 2.04 µs | ~490 тыс. разборов/с |
| Mega RAG | `locate_centroid_cluster` (50 тыс. узлов, mmap CSR) | 7.5 ns | ~130 млн запросов/с |
| Thought Graph | `query_path_binup` (BINUP LCA, 64 узла) | 22 ns | ~45 млн запросов/с |
| Thought Graph | `reroot` (перестройка `up`/`depth`) | 2.2 µs | ~450 тыс. операций/с |
| Dinitz Decoder | сброс по маске + `max_flow` (64 вершины, 192 дуги) | 780 ns | ~1.28 млн операций/с |
| Pipeline | холодный старт `AEEnginePipeline(50000, 6)` | 45 ms | — |
| Pipeline | `infer`, поле `latency_us` (ядро конвейера) | 19.5 µs (p95 23 µs) | ~51 тыс. запросов/с |
| Pipeline | `infer`, полное время с Time-Ray синхронно | 22.5 µs (p95 28 µs) | ~44 тыс. запросов/с |
| Pipeline | `infer`, полное время с Time-Ray через `std::async` | 51 µs (p95 85 µs) | ~20 тыс. запросов/с |

Память и размеры на том же прогоне:

| Метрика | Значение |
|---|---:|
| RSS процесса после всех стадий | 33.8 MB |
| CSR-представление Mega RAG (50 тыс. узлов, степень 6) | 3.6 MB |
| Топология графа мыслей | 64 узла, 96 рёбер (192 направленные дуги) |
| AST для тестового запроса | 451 токен |
| `max_flow` / `path_bottleneck` для тестового запроса | 259 / 120 |

Что важно при чтении цифр:

- `latency_us` измеряется внутри `infer` до запуска Time-Ray и компиляции AST,
  поэтому полное время выше на 3–4 µs (синхронный вариант) и примерно на
  30 µs больше при `std::async`: доминирует создание потока, а не сам расчёт.
- `locate_centroid_cluster` — это подъём по `centroid_parent` (не более 4 шагов)
  по уже прогретому индексу; значение показывает стоимость указателей, а не
  поиск по холодному mmap.
- Русский парсер примерно в 8 раз медленнее английского на этих строках из-за
  большего числа слов и более длинных морфемных цепочек, а не из-за словаря.
- Тесты после сборки: `ctest --test-dir build --output-on-failure` — 2/2
  (C++23 ядро и Python-обвязка), ~0.3 с.

## Общий интерфейс языков

`ae::lang::LanguageInterface` — единый интерфейс `parse`, `language` и `family`;
результат содержит морфемы (для human-языков) и общий `MeaningNode` для
древовидного представления. Один интерфейс доступен из пространств имён
`ae::lang::human`, `ae::lang::science`, `ae::lang::verbal` и
`ae::lang::programing`. Программный парсер строит дерево программы, функций,
блоков, объявлений, управляющих конструкций и лексических узлов.

`ae::lang::select_parser(family, text)` выбирает парсер по семейству; для
`SCIENCE` и `VERBAL` возвращается `nullptr`, пока эти семейства не получат
реализацию. Одноаргументный `select_parser(text)` остаётся совместимым
маршрутизатором русского/английского human-парсера.

## Морфемный инвентарь и 64-битная маска сочетаемости

`ae::mind::MorphemicPatriciaTrie` заполняется при конструировании из двух
встроенных инвентарей — русского (`seed_russian_inventory`) и английского
(`seed_english_inventory`):

| Категория | Русский | Английский | Примеры |
|---|---:|---:|---|
| Приставки | 146 | 73 | `пере-`, `недо-`, `сверх-` / `neuro-`, `multi-`, `trans-` |
| Корни (20 кластеров) | 737 | 652 | `граф`, `мысл`, `поток` / `graph`, `neuron`, `knowledge` |
| Суффиксы | 76 | 44 | `-ирование`, `-тельность` / `-ization`, `-ology` |
| Окончания и форманты | 69 | 7 | `-ого`, `-ться` / `-ing`, `-ed`, `-ly` |

Всего **1804** морфемы (2580 узлов Patricia-триe), из них 1028 русских и 776
английских. Корни разложены по кластерам `GRAPH`, `KNOW`, `FLOW`, `TEXT`,
`NUM`, `TIME`, `SPACE`, `MIND`, `COMPUTE`, `NET`, `PHYS`, `CHEM`, `BIO`,
`MED`, `ECON`, `LAW`, `SOC`, `QUALITY`, `NATURE`, `ACTION`; английские корни
используют те же кластеры, поэтому маска сочетаемости двуязычна.

Биты маски:

```text
[ 0.. 7]  базовые домены смысла (semantic_tag 0..7 декодера Диница)
[ 8..15]  домены, гейтируемые декодером по tag 8..15
[16..23]  интент-биты Taylor-Intent Kernel
[24..27]  тип морфемы: PREFIX / ROOT / SUFFIX / INFLECTION
[28..47]  семантический кластер корня
[48..63]  свободно для регистрации через C ABI (`ae_engine_add_morpheme`)
```

Идентификаторы `0..255` зарезервированы под динамическую регистрацию из
Python/Rust/Go, встроенный инвентарь занимает `256+`, поэтому вызовы
`ae_engine_add_morpheme` не перетирают встроенные морфемы.

`compile_morpheme_mask` работает пословно с жадным longest-match и непрерывной
цепочкой морфем: внутри слова выбирается самая длинная зарегистрированная
морфема, разбор продолжается с её конца, а неизвестный участок обрывает
цепочку. Поэтому вложенные совпадения (`аст` внутри `астро`) не завышают
`matched_morphemes`, а слово вне словаря (`кот`) не даёт ложную приставку
`от`.

Русский парсер (`ae::lang::ru::Parser`) использует тот же инвентарь приставок и
аффиксов. Короткие неоднозначные приставки (`ко-`, `во-`, `су-`, `ре-`, `ни-`,
`де-`) исключены из обоих слоёв — без словаря основ они разрушают частотные
слова (`дерево`, `вопрос`, `компьютер`). Дополнительно парсер отделяет не более
двух приставок и только если остаток основы не короче трёх символов
(`detail::MIN_STEM_BYTES`, `detail::MAX_PREFIXES_PER_WORD`).

## Топология графа мыслей и поток Диница

`AEEnginePipeline` строит `thought_graph_` и `decoder_` из **одного** списка
рёбер (`std::vector<ThoughtGraphTopology::Edge> topology`), поэтому
последовательность `optimal_path`, которую печатает AST, — это настоящий путь
в остовном дереве: `parent_id`, `depth` и LCA в `THOUGHT_NODE` согласованы с
рёбрами, по которым считается поток.

Подграф состоит из 64 узлов:

- **10 параллельных цепочек** `root → sink`. Инвариантная цепочка размечена
  тегами `0..4` (входят в `BASE_MASK`) и существует при любой маске; остальные
  девять размечены тегом `28 + cluster` и открываются только кластерами,
  реально встретившимися в запросе (`GRAPH`, `KNOW`, `FLOW`, `TEXT`, `NUM`,
  `COMPUTE`, `MIND`, `NET`, `ACTION`);
- **ветви** оставшихся 22 узлов с кластерными тегами;
- **24 хорды**, создающие циклы и дополнительные пути для потока.

Итог: 96 рёбер (192 направленных записи при лимите `DinitzDecoder::MAX_E =
1024`). Пропускная способность цепочки убывает к стоку, поэтому

- `max_flow` зависит от запроса: `120` для пустого/неизвестного слова (только
  инвариантная цепочка), `141` для `граф`, `210` для `поток знаний`, `214` для
  `переподключение графа мыслей`;
- `path_bottleneck` — истинный минимум веса на пути в остовном дереве
  (`query_path_min_binup`), а не максимум;
- поле `path_max_weight` в AST по-прежнему показывает накопленный максимум
  (`query_path_binup`), то есть верхнюю оценку веса пути.

Путь, попадающий в AST, всегда идёт по инвариантной цепочке
`0 → 8 → 21 → 38 → 52 → 63`: это гарантирует непустой объяснимый след при
любой маске, а семантические кластеры запроса влияют на агрегированный
`max_flow` и на пропускную способность параллельных ветвей. `reroot(node)`
перестраивает `up`/`depth`/LCA относительно новой вершины.

## Выход на естественном языке и обратный разбор

`AEEnginePipeline::infer` дополнительно к AST заполняет
`InferenceResult::natural_language` (`NaturalLanguageReport`). Модуль
`natural_language_output.hpp` строит короткий ответ на языке запроса из
подтверждённых морфемной маской доменов и интент-вектора, после чего
**возвращает сгенерированный текст в тот же парсер** (`ru`/`en`) и в
морфемный trie.

Контур работает так:

1. из маски запроса извлекаются семантические кластеры (биты `28..47`);
2. домены сортируются по значению соответствующего интента и переводятся в
   слово, построенное из зарегистрированного корня (`граф`, `знание`, `поток`,
   `нейрон`, …);
3. текст разбирается обратно: `lang::select_parser(text)->parse(text)` плюс
   `compile_morpheme_mask`;
4. считается `cluster_recall` (не потеряли ли смысл) и `cluster_precision`
   (не добавили ли лишние домены);
5. если полнота меньше единицы, вербализатор добавляет недостающий домен и
   повторяет разбор — до полного совпадения или исчерпания доменов.

Порог сходимости: `recall ≥ 0.99` и `precision ≥ 0.8`. Служебные морфемы
(PREFIX/SUFFIX/INFLECTION, кластер `MORPH`, бит 48) в сверке не участвуют,
поэтому окончания не считаются потерей смысла. Числовая часть ответа
(`max_flow/bottleneck`) состоит только из цифр и не добавляет морфем.

Пример для запроса `нейроны графа знаний и поток данных`:

```text
text:      Тема: нейрон, граф, поток и знание. 249/120
recall=1.00 precision=1.00 morphemes=12 passes=1 consistent=true
feedback:  Обратный разбор: язык=ru, морфем=12, полнота=1.00, точность=1.00, проходов=1.
```

Доступ из FFI и Python:

- C ABI: `ae_engine_get_last_nl`, `ae_engine_get_last_nl_feedback`, поля
  `nl_cluster_recall`, `nl_cluster_precision`, `nl_reparsed_morphemes`,
  `nl_refinement_passes`, `nl_consistent` в `AEInferenceMetrics`;
- Python: `AEInferenceOutput.nl_text`, `.nl_feedback`, `.nl_consistent` и др.;
- `AESLMOrchestrator.answer` использует `nl_text` как основной
  `verbalized_response`, а детальный отчёт оставляет в `trace_response`.

Важно: `latency_us` по-прежнему измеряется до Time-Ray, AST и обратного
разбора, поэтому вербализация не искажает заявленную задержку ядра.

## Обучаемый графовый контур и предметные метрики

Помимо детерминированного ядра в проект добавлен обучаемый контур, который
использует алгоритмы как источник признаков:

- `include/algorithm_features.hpp` — 115-мерный вектор из trie-маски, счётчиков и
  счётчиков кластеров, потока Диница, центроидной декомпозиции, MST/LCA-пути,
  Time-Ray и NL round-trip;
- `tools/make_text_cluster_synthetic.py` — детерминированная синтетическая
  волна, проверяемая trie и попадающая только в `train`;
- `training/ae_features.py` — сборка датасета из корпуса запросов;
- `training/ae_graph_train.py` — GraphSAGE-подобный классификатор с явным
  backward pass, Adam, early stopping и стратифицированной кросс-валидацией;
- `training/ae_graph_hip.py` — HIP smoke test для Vega 7;
- `benchmarks/logic_tasks.py` — проверка алгоритмов против brute-force;
- `benchmarks/text_tasks.py` — метрики анализа текста.

Быстрый запуск:

```bash
python3 tools/make_text_cluster_corpus.py
python3 tools/make_text_cluster_synthetic.py
python3 tools/merge_corpora.py training/corpora/text_clusters.json \
  training/corpora/text_clusters_manual.json \
  training/corpora/text_clusters_manual_v2.json \
  training/corpora/text_clusters_synthetic.json \
  -o training/corpora/text_clusters_expanded.json
python3 -m training.ae_features training/corpora/text_clusters_expanded.json \
  data/text_clusters_expanded_features.json
python3 -m training.ae_graph_train data/text_clusters_expanded_features.json --cv 5 \
  --epochs 400 --hidden-dim 0 --learning-rate 0.01 --weight-decay 0.01
python3 -m benchmarks.run_all --output reports/benchmarks.json
```

Vega 7 (`gfx90c`, XNACK enabled):

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 -m training.ae_graph_hip --target gfx900
```

Подробности, формат датасета, раскладка признаков и измеренные метрики —
в [TRAINING.md](TRAINING.md).

## Ручная разметка корпуса

Seed-корпус (90 запросов) отделён от ручной разметки
(`training/corpora/text_clusters_manual.json` и
`text_clusters_manual_v2.json`, по 180 запросов) и синтетической волны
(`text_clusters_synthetic.json`, 4 500 запросов только в `train`).
Объединённый корпус собирается `tools/merge_corpora.py`, проверяется
`tests/test_corpus.py` и описывается в [LABELING.md](LABELING.md): правила
классов, баланс split'ов, текущие метрики и следующие шаги.

## Архитектурные решения (ADR)

Обоснование каждого алгоритма и технического решения вынесено в
[adr/README.md](adr/README.md). Каждый ADR содержит контекст, принятое
решение, отклонённые альтернативы и последствия.
