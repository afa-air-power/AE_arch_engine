#!/usr/bin/env python3
"""Build the semantic-cluster text corpus used by the feature pipeline.

Labels follow the thought-graph chain order:
0 GRAPH, 1 KNOW, 2 FLOW, 3 TEXT, 4 NUM, 5 COMPUTE, 6 MIND, 7 NET, 8 ACTION.

ADR: docs/adr/0012-trainable-graph-contour.md
"""

from __future__ import annotations

import argparse
import json
import pathlib

CLUSTERS = {
    "GRAPH": [
        "граф знаний и дерево вершин",
        "структура связей между узлами",
        "остовное дерево и компоненты связности",
        "иерархия кластеров и подграф",
        "разреженный граф и матрица смежности",
        "graph structure and spanning tree",
        "матрица смежности и степень вершины",
        "обход графа в глубину и в ширину",
        "adjacency matrix and vertex degree",
        "graph traversal and connected components",
    ],
    "KNOW": [
        "знание смысл и память",
        "онтология понятий и фактов",
        "контекст и семантика данных",
        "истина гипотеза и логика",
        "категория концепт и определение",
        "knowledge ontology and semantic facts",
        "понятие и определение термина",
        "база знаний и правила вывода",
        "semantic meaning and context",
        "knowledge base and inference rules",
    ],
    "FLOW": [
        "поток данных и маршрут",
        "конвейер обработки и этапы",
        "каскад процессов и очередь",
        "пропускная способность канала",
        "итерация цикла и переход",
        "data flow pipeline and routing",
        "очередь задач и планировщик",
        "балансировка нагрузки канала",
        "scheduler queue and buffer",
        "stream processing stages",
    ],
    "TEXT": [
        "текст и морфемы слова",
        "токены лексемы и предложения",
        "язык грамматика синтаксис",
        "корпус документов и перевод",
        "алфавит символ и пунктуация",
        "text tokens and morphology",
        "морфологический разбор слова",
        "предложение и пунктуация",
        "corpus of documents and translation",
        "grammar syntax and punctuation",
    ],
    "NUM": [
        "число вектор и матрица",
        "полином тейлора и градиент",
        "алгоритм вычисления метрики",
        "вероятность и статистика",
        "интеграл предел и производная",
        "number vector and matrix",
        "сумма ряда и предел",
        "матричное умножение и спектр",
        "probability distribution and statistics",
        "integral derivative and limit",
    ],
    "COMPUTE": [
        "программа процессор и модуль",
        "сборка компиляция и код",
        "вычислительная архитектура",
        "память регистры и кэш",
        "библиотека интерфейс и платформа",
        "program processor and module",
        "компилятор и байткод",
        "кэш регистров и инструкции",
        "compiler bytecode and instruction",
        "operating system kernel and driver",
    ],
    "MIND": [
        "мозг нейрон и сознание",
        "когнитивная модель разума",
        "интеллект и обучение",
        "психика восприятие и внимание",
        "мотивация решение и прогноз",
        "brain neuron and cognition",
        "внимание память и решение",
        "эмоция и поведение",
        "attention memory and decision",
        "emotion behavior and perception",
    ],
    "NET": [
        "сеть протокол и маршрутизация",
        "обмен пакетами между узлами",
        "соединение и топология сети",
        "распределённая сеть и трафик",
        "сокет адрес и порт",
        "network protocol and routing",
        "маршрутизатор и шлюз",
        "сессия и авторизация клиента",
        "router gateway and traffic",
        "session authorization and socket",
    ],
    "ACTION": [
        "действие операция преобразование",
        "запуск остановка и повтор",
        "изменение состояния системы",
        "переход и завершение",
        "сортировка фильтрация и агрегация",
        "action operation and transformation",
        "поиск и сортировка данных",
        "генерация и фильтрация",
        "search sort and filter",
        "generation and transformation",
    ],
}

SPLIT_BY_INDEX = ["train", "train", "train", "val", "train", "test",
                  "train", "train", "val", "test"]


def build_samples() -> list[dict]:
    samples = []
    for label, (name, queries) in enumerate(CLUSTERS.items()):
        for index, query in enumerate(queries):
            samples.append({
                "query": query,
                "label": label,
                "cluster": name,
                "split": SPLIT_BY_INDEX[index % len(SPLIT_BY_INDEX)],
            })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="training/corpora/text_clusters.json")
    args = parser.parse_args()
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "text_clusters",
        "labels": list(CLUSTERS),
        "samples": build_samples(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(payload['samples'])} samples to {path}")


if __name__ == "__main__":
    main()
