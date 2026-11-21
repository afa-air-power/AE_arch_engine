# Обучаемый графовый контур и предметные метрики

Документ описывает полный контур: признаки из алгоритмов ядра, сборку
датасета, обучение, оценку качества и HIP-проверку Vega 7.

## 1. Принцип

Алгоритмы ядра не заменяются нейросетью. Они поставляют признаки и проверяют
результат:

| Модуль | Что даёт модели |
|---|---|
| Taylor-Intent Kernel | 16-мерный intent-вектор |
| Morphemic Patricia Trie | popcount маски, matched-морфемы, 20 индикаторов и 20 счётчиков кластеров, детерминированные составные термины |
| Mega RAG (centroid decomposition) | id центроида, число шагов, глубина |
| Dinitz max-flow | `max_flow`, `path_bottleneck`, отношения |
| Thought Graph (Kruskal + BINUP LCA) | длина пути, веса рёбер, степени MST/хорд, монотонность |
| Time-Ray | forward/backward scores, число отсечённых аномалий |
| NL round-trip | cluster recall/precision, число проходов |

Вектор признаков — 118 значений в `[0, 1]` (`include/algorithm_features.hpp`). Последние 22 признака описывают структуру предложения: поддержку у голов чанков, максимум поддержки по локальным поддеревьям, глубину дерева и RDT-повороты.
Раскладка:

```text
[ 0..15] intent
[16..24] морфемная маска и словарь
[25..27] центроид
[28..31] поток Диница
[32..41] топология графа мыслей
[42..45] Time-Ray + средний вес ребра пути
[46..50] обратный разбор NL
[51..55] метаданные запроса
[56..75] индикаторы 20 семантических кластеров
[76..95] поддержка кластеров: log1p(число морфем) / log1p(16)
```

Индикатор отвечает на вопрос «тема встретилась», счётчик — «насколько
сильно она поддержана». Это различает запросы с одинаковым набором битов,
например `graph coloring` и `graph spectrum`. Дополнительно trie содержит
небольшую таблицу составных терминов (`neural network → MIND`,
`routing table → NET`, `network flow → FLOW`): она добавляет поддержку
главному кластеру фразы, но не отключает морфемный разбор.

Признаки детерминированы: время выполнения в вектор не входит, поэтому два
одинаковых запроса дают побитово одинаковый вектор (проверяется тестом).

## 2. Группы признаков

`intent`, `centroid` и `timeray` выводятся из байтового хэша запроса и кодируют
его идентичность, а не смысл. На малой выборке они дают запоминание вместо
обобщения, поэтому по умолчанию исключены. Текущий default — 31 признак:

```text
cluster_core, cluster_counts_core   # 9 битов + 9 счётчиков
chunk_core, sentence                # 9 локальных максимумов + глубина дерева
rdt                                  # повороты, глубина голов, глубина RDT-дерева
```

RDT-представление строится обратимыми локальными поворотами исходного дерева.
Исходные `parent/depth` сохраняются, а `normalized_parent/normalized_depth`
образуют второе дерево. История операций хранится в `rotations`.

Разбор предложения в C++ сохраняет токены, чанки и связи «чанк → голова →
зависимые слова». Благодаря этому модель видит не только набор слов всего
запроса, но и локальное поддерево. Например, в `routing table` голова
получает поддержку `NET`, а в `network flow` — `FLOW`. Пунктуация и союзы
создают границы чанков, поэтому независимые части предложения не смешиваются
в одну локальную структуру.

`cluster_core` — только кластеры, совпадающие с таксономией меток
(`graph, know, flow, text, num, mind, compute, net, action`). Остальные 11
кластеров почти всегда нулевые и только добавляют дисперсию.

Полный набор доступен через `--groups`:

```bash
python3 -m training.ae_features corpus.json data/all.json \
  --groups intent,morphology,centroid,flow,topology,timeray,nl,query,clusters,cluster_counts
```

## 3. Формат датасета

```json
{
  "node_features": [[0.1, 0.2]],
  "adjacency": [],
  "labels": [0, 1],
  "split": ["train", "test"],
  "queries": ["граф и дерево"],
  "feature_names": ["cluster_graph", "..."],
  "meta": {"feature_dim": 18, "feature_groups": ["..."], "graph": "none"}
}
```

`node_features` — `[N, F]`, `labels` — целые классы, `split` —
`train` / `val` / `test` для каждого узла. При `"graph": "none"` поле
`adjacency` пустое (`0×0`), а не плотная `[N, N]`: обучение трактует пустой
граф как тождественную агрегацию и не тратит память на матрицу 4 950 × 4 950.

## 4. Граф над образцами

`--graph` задаёт структуру связей между образцами:

- `none` — без рёбер (измеренно лучший режим; кодируется пустым adjacency);
- `lexical` — рёбра по пересечению словаря запросов (Jaccard ≥ `--min-similarity`);
- `knn` — k ближайших соседей по стандартизованным признакам (`--neighbors`).

Граф должен кодировать реальные отношения. На текстовом корпусе и `lexical`,
и `knn` смешивают темы и ухудшают метрику, поэтому по умолчанию выбран
`none`; для корпуса с явными парафразами или документами используйте
`lexical`.

## 5. Обучение

```bash
# 1. corpus: seed (90) + two manual waves (360) + synthetic train (4 500)
python3 tools/make_text_cluster_corpus.py
python3 tools/make_text_cluster_synthetic.py
python3 tools/merge_corpora.py training/corpora/text_clusters.json \
  training/corpora/text_clusters_manual.json \
  training/corpora/text_clusters_manual_v2.json \
  training/corpora/text_clusters_synthetic.json \
  -o training/corpora/text_clusters_expanded.json

# 2. признаки из алгоритмов ядра (18-мерный default)
python3 -m training.ae_features training/corpora/text_clusters_expanded.json \
  data/text_clusters_expanded_features.json

# 3. обучение с фиксированным split ручного якоря
python3 -m training.ae_graph_train data/text_clusters_expanded_features.json \
  --epochs 400 --hidden-dim 0 --learning-rate 0.01 --weight-decay 0.01 \
  --checkpoint checkpoints/text_clusters_expanded.json

# 4. стратифицированная кросс-валидация (надёжнее одного test split)
python3 -m training.ae_graph_train data/text_clusters_expanded_features.json \
  --cv 5 --epochs 400 --hidden-dim 0 --learning-rate 0.01 --weight-decay 0.01
```

Модель — двухслойный GraphSAGE-подобный классификатор с mean-агрегацией и
явным backward pass. `--hidden-dim 0` включает линейный режим (устойчивее на
малых корпусах). Используются Adam, L2-регуляризация, стандартизация по train
split и early stopping по validation macro-F1 (восстанавливается состояние
лучшей эпохи, а не последней).

### Измеренные метрики seed-корпуса

| Метрика | Значение |
|---|---:|
| Hold-out test accuracy (18 образцов) | 0.667 |
| Hold-out test macro-F1 | 0.663 |
| 5-fold CV accuracy | **0.856 ± 0.125** |
| 5-fold CV macro-F1 | **0.844 ± 0.141** |

Корпус маленький (54 train / 18 val / 18 test), поэтому один test split шумит;
кросс-валидация — основная оценка. Для роста качества нужен размеченный корпус
на сотни запросов на кластер, а не усложнение модели.

### Объединённый корпус: ручной якорь и синтетика

Seed-корпус (90 запросов) дополнен двумя волнами ручной разметки
`text_clusters_manual.json` и `text_clusters_manual_v2.json` (по 180
запросов) и детерминированной синтетической волной
`text_clusters_synthetic.json` (4 500 запросов, 500 на класс).
Объединённый корпус `training/corpora/text_clusters_expanded.json`
содержит 4 950 запросов: split `4770 train / 90 val / 90 test`
(по 530/10/10 на класс). Вся синтетика имеет `split: "train"`, поэтому
ручной `val`/`test` остаётся чистым якорем.

Синтетический генератор проверяет каждый запрос через
`compile_morpheme_analysis`: метка принимается, только если нужный кластер
присутствует в trie-разборе, а для трудных пар — если он доминирует.
Поэтому синтетика не создаёт признаков, которых ядро не умеет выдавать.

Правила и цикл разметки — в [LABELING.md](LABELING.md).

| Метрика | Seed (90) | Ручной v2 (450) | + синтетика (4 950) |
|---|---:|---:|---:|
| Hold-out test accuracy | 0.667 (18) | 0.833 (90) | **0.936 ± 0.011** (90) |
| Hold-out test macro-F1 | 0.663 | 0.835 | **0.935 ± 0.011** |
| 5-fold CV accuracy, ручной якорь | 0.856 ± 0.125 | 0.809 ± 0.039 | **0.879 ± 0.009** |
| 5-fold CV macro-F1, ручной якорь | 0.844 ± 0.141 | 0.808 ± 0.037 | **0.878 ± 0.009** |
| 5-fold CV accuracy, полный корпус | — | — | **0.969 ± 0.002** |

CV-числа для нового корпуса — среднее по 5 seed разбиения. Hold-out на
ручном якоре (`val` + `test`, 180 примеров) даёт **0.899 ± 0.006**;
`val` — 0.862 ± 0.009, `test` — 0.936 ± 0.011. Разбивка:

| Подмножество | Примеров | Accuracy |
|---|---:|---:|
| Однозначные (один кластерный бит) | 121 | **0.955 ± 0.000** |
| Трудные (два и более бита) | 59 | **0.816 ± 0.031** |
| RU | 81 | 0.844 ± 0.009 |
| EN | 99 | 0.943 ± 0.016 |

Трудные пары выросли с `0.500` до `0.813` за счёт счётчиков кластеров и
слоя составных терминов. Остаточные ошибки — неизвестные контексты, а не
отдельные корни; следующий шаг — синтаксические признаки.

## 6. Предметные метрики

### Логические задачи для ядра

`benchmarks/logic_tasks.py` сравнивает каждый алгоритм с независимой
brute-force реализацией:

| Задача | Что проверяется |
|---|---|
| `path_matches_bfs` | `path_between` == BFS-путь в MST |
| `path_max_weight_matches_bruteforce` | `query_path_binup` == max вес на пути |
| `path_min_weight_matches_bruteforce` | `query_path_min_binup` == истинный bottleneck |
| `reroot_preserves_paths` | reroot не меняет путь и его веса |
| `kruskal_maximum_spanning_tree` | n−1 рёбер, вес == эталонному Краскалу |
| `dinitz_matches_edmonds_karp` | max-flow == Edmonds-Karp на случайных графах |
| `centroid_lookup_bounds` | hops ≤ 4, глубина ≤ ⌈log₂N⌉+1 |
| `trie_known_root_clusters` | известные корни дают ожидаемый кластер |
| `trie_unknown_words_clean` | неизвестные слова не дают ложных кластеров |
| `flow_bottleneck_invariants` | bottleneck ≤ flow, путь root→sink, flow > 0 |

### Задачи анализа текста

`benchmarks/text_tasks.py`:

| Задача | Что проверяется |
|---|---|
| `language_detection` | RU/EN определяется верно |
| `semantic_cluster_detection` | precision/recall/F1 по кластерам запроса |
| `morpheme_segmentation` | ожидаемая основа присутствует в разборе |
| `unknown_words_no_clusters` | мусорные слова не создают кластеров |
| `nl_round_trip_consistency` | recall ≥ 0.99, отчёт согласован |
| `deterministic_output` | одинаковый вход → одинаковый выход |
| `ast_contract` | маркеры AST и согласованность с `optimal_path` |

Запуск:

```bash
python3 -m benchmarks.logic_tasks
python3 -m benchmarks.text_tasks
python3 -m benchmarks.run_all --output reports/benchmarks.json
```

Текущий результат на `gfx90c`/CPU: **1.000** по обоим наборам (см.
`reports/benchmarks.json`).

## 7. Vega 7 / HIP

Стенд: AMD Ryzen 5 5500U, Radeon Graphics (Vega 7), `gfx90c`, XNACK enabled,
ROCm 7.16.

Нативный `gfx90c` code object аварийно завершается в runtime с проверкой
XNACK. Рабочая конфигурация:

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.0 python3 -m training.ae_graph_hip --target gfx900
```

Скрипт применяет override по умолчанию. Проверяются: компиляция kernel,
выделение памяти, H2D-копирование, dispatch, D2H-копирование и корректность
данных. Отключить override: `--hsa-override ""`.

Регистрация в CTest:

```bash
cmake -S . -B build -DAE_BUILD_PYBIND11=ON -DAE_ENABLE_HIP_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Текущее состояние: HIP smoke test проходит на `gfx900`. Полноценный HIP-backend
обучения (tensor-операции на GPU) пока не реализован; CPU-контур и проверка
устройства работают независимо.

## 8. Полная проверка

```bash
cmake -S . -B build -DAE_BUILD_PYBIND11=ON -DAE_ENABLE_HIP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Ожидается 7/7 тестов: ядро C++, Python-обвязка, контракт корпуса, контракт
обучения, признаки, предметные бенчмарки, HIP smoke test.
