# Обучаемый графовый контур

Проект сохраняет алгоритмы графа как часть признакового и проверочного контура. `Dinitz`, centroid-поиск, MST/LCA, bottleneck и путь объяснения не заменяются нейросетью. Они могут быть рассчитаны при подготовке графа и переданы модели как node/edge features, а путь и поток используются для последующей проверки результата.

## Формат датасета

Файл JSON содержит:

```json
{
  "node_features": [[0.1, 0.2]],
  "adjacency": [[0, 1], [1, 0]],
  "labels": [0, 1],
  "split": ["train", "test"]
}
```

`node_features` имеет форму `[N, F]`, `adjacency` имеет форму `[N, N]`, `labels` содержит целочисленные классы, а `split` задаёт `train`, `val` или `test` для каждого узла.

## CPU baseline

```bash
python3 tools/make_demo_dataset.py data/demo.json
python3 -m training.ae_graph_train data/demo.json \
  --epochs 100 \
  --hidden-dim 32 \
  --checkpoint checkpoints/ae_graph.json
python3 tests/test_graph_training.py
```

Модель является детерминированным NumPy GraphSAGE-baseline с двумя слоями агрегации и явным backward pass. Checkpoint содержит веса и историю обучения.

## Vega 7 / HIP

На текущем стенде ROCm обнаруживает GPU как `gfx90c`, Radeon Graphics, XNACK enabled. Smoke test компилирует kernel под `gfx90c:xnack+` и проверяет выделение памяти, копирование, dispatch и возврат данных:

```bash
python3 -m training.ae_graph_hip
```

Если runtime завершает процесс при загрузке code object с ошибкой `isCompatible` по XNACK, это означает несовместимость установленного ROCm runtime/toolchain с агентом GPU. Обучение в таком состоянии нельзя считать GPU-обучением. Нужно согласовать версию ROCm и режим XNACK, затем повторить smoke test до интеграции GPU backend в trainer.

## Следующий обязательный слой

Перед заявлением о точности нужно заменить demo dataset реальным размеченным графом, зафиксировать правила split и добавить метрики предметной задачи. Текущий demo dataset предназначен для проверки формата, checkpointing и воспроизводимости.
