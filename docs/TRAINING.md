# Обучаемый графовый контур и предметные метрики

Документ описывает полный контур: признаки из алгоритмов ядра, сборку
датасета, обучение, оценку качества и HIP-проверку Vega 7.

## 1. Принцип

Алгоритмы ядра не заменяются нейросетью. Они поставляют признаки и проверяют
результат:

| Модуль | Что даёт модели |
|---|---|
| Taylor-Intent Kernel | 16-мерный intent-вектор |
| Morphemic Patricia Trie | popcount маски, matched-морфемы, 20 индикаторов кластеров |
| Mega RAG (centroid decomposition) | id центроида, число шагов, глубина |
| Dinitz max-flow | `max_flow`, `path_bottleneck`, отношения |
| Thought Graph (Kruskal + BINUP LCA) | длина пути, веса рёбер, степени MST/хорд, монотонность |
| Time-Ray | forward/backward scores, число отсечённых аномалий |
| NL round-trip | cluster recall/precision, число проходов |

Вектор признаков — 76 значений в `[0, 1]` (`include/algorithm_features.hpp`).
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
```

Признаки детерминированы: время выполнения в вектор не входит, поэтому два
одинаковых запроса дают побитово одинаковый вектор (проверяется тестом).

## 2. Группы признаков

`intent`, `centroid` и `timeray` выводятся из байтового хэша запроса и кодируют
его идентичность, а не смысл. На малой выборке они дают запоминание вместо
обобщения, поэтому по умолчанию исключены:

```text
morphology, flow, topology, nl, query, clusters   # 53 признака
```

Полный набор доступен через `--groups`:

```bash
python3 -m training.ae_features corpus.json data/all.json \
  --groups intent,morphology,centroid,flow,topology,timeray,nl,query,clusters
```

## 3. Формат датасета

```json
{
  "node_features": [[0.1, 0.2]],
  "adjacency": [[0, 1], [1, 0]],
  "labels": [0, 1],
  "split": ["train", "test"],
  "queries": ["граф и дерево"],
  "feature_names": ["mask_popcount", "..."],
  "meta": {"feature_dim": 53, "feature_groups": ["..."], "graph": "none"}
}
```

`node_features` — `[N, F]`, `adjacency` — `[N, N]`, `labels` — целые классы,
`split` — `train` / `val` / `test` для каждого узла.

## 4. Граф над образцами

`--graph` задаёт структуру связей между образцами:

- `none` — без рёбер (измеренно лучший режим на seed-корпусе);
- `lexical` — рёбра по пересечению словаря запросов (Jaccard ≥ `--min-similarity`);
- `knn` — k ближайших соседей по стандартизованным признакам (`--neighbors`).

Граф должен кодировать реальные отношения. На корпусе из 90 запросов и
`lexical`, и `knn` смешивают темы и ухудшают метрику, поэтому по умолчанию
выбран `none`; для корпуса с явными парафразами или документами используйте
`lexical`.

## 5. Обучение

```bash
# 1. корпус (seed, 90 запросов, 9 кластеров)
python3 tools/make_text_cluster_corpus.py

# 2. признаки из алгоритмов ядра
python3 -m training.ae_features training/corpora/text_clusters.json \
  data/text_clusters_features.json

# 3. обучение с фиксированным split
python3 -m training.ae_graph_train data/text_clusters_features.json \
  --epochs 300 --hidden-dim 8 --learning-rate 0.01 --weight-decay 0.01 \
  --patience 60 --checkpoint checkpoints/text_clusters.json

# 4. стратифицированная кросс-валидация (надёжнее одного test split)
python3 -m training.ae_graph_train data/text_clusters_features.json \
  --cv 5 --epochs 300 --hidden-dim 8 --learning-rate 0.01 --weight-decay 0.01
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

Ожидается 6/6 тестов: ядро C++, Python-обвязка, контракт обучения, признаки,
предметные бенчмарки, HIP smoke test.
