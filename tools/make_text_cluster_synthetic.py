#!/usr/bin/env python3
"""Generate a large synthetic corpus for the semantic-cluster classifier.

The corpus is deterministic and validated against the same Morphemic Patricia
Trie that produces the runtime features.  A candidate is kept only when the
intended cluster is actually present in the trie mask; this prevents synthetic
labels from drifting away from the feature representation.

Splits: every synthetic sample is assigned to ``train``.  The manual waves
(``text_clusters_manual*.json``) remain the held-out evaluation anchor, so
adding synthetic data cannot leak into ``val``/``test``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ["GRAPH", "KNOW", "FLOW", "TEXT", "NUM", "COMPUTE", "MIND", "NET", "ACTION"]
# Позиция кластера в 20-битном векторе trie (Semantic enum).
CLUSTER_INDEX = {
    "graph": 0, "know": 1, "flow": 2, "text": 3, "num": 4,
    "mind": 7, "compute": 8, "net": 9, "action": 19,
}

# Candidate inventories.  Terms were chosen so that the trie attaches them to
# the intended cluster; the generator still validates every final sentence.
LEXICON: dict[str, dict[str, dict[str, list[str]]]] = {
    "GRAPH": {
        "ru": {
            "subjects": [
                "граф", "дерево", "вершина", "ребро", "подграф", "остовное дерево",
                "компонента связности", "обход графа", "матрица смежности",
                "топологическая сортировка", "двудольный граф", "изоморфизм графов",
                "ациклический граф", "эйлерова цепь", "мост графа", "гиперграф",
                "раскраска графа", "плотность графа", "диаметр дерева", "центр графа",
                "степень вершины", "матрица инцидентности", "связность графа",
                "разрез графа",
            ],
            "actions": [
                "поиск пути", "обход вершин", "топологическая сортировка",
                "раскраска вершин", "построение остова", "проверка связности",
                "поиск мостов", "разбор дерева", "анализ смежности", "поиск цикла",
            ],
            "qualifiers": [
                "связный", "ориентированный", "взвешенный", "двудольный",
                "разреженный", "плотный", "ациклический", "планарный",
            ],
        },
        "en": {
            "subjects": [
                "graph", "tree", "vertex", "edge", "subgraph", "spanning tree",
                "connected component", "graph traversal", "adjacency matrix",
                "directed acyclic graph", "bipartite graph", "graph isomorphism",
                "hypergraph", "graph coloring", "graph density", "tree diameter",
                "graph center", "vertex degree", "edge cut", "graph connectivity",
                "adjacency list", "graph minor", "tree decomposition", "graph automorphism",
            ],
            "actions": [
                "path search", "vertex traversal", "topological sorting",
                "vertex coloring", "spanning tree construction", "connectivity check",
                "bridge search", "tree decomposition", "adjacency analysis", "cycle search",
            ],
            "qualifiers": [
                "connected", "directed", "weighted", "bipartite",
                "sparse", "dense", "acyclic", "planar",
            ],
        },
    },
    "KNOW": {
        "ru": {
            "subjects": [
                "знание", "онтология", "логика", "семантика", "понятие", "смысл",
                "контекст", "рассуждение", "база знаний", "представление знаний",
                "экспертная система", "гипотеза", "правило вывода", "логическое следствие",
                "тезаурус", "категоризация", "концепт", "определение", "аргументация",
                "достоверность", "противоречие", "семантическая сеть", "извлечение знаний",
                "модель знаний",
            ],
            "actions": [
                "логический вывод", "проверка гипотезы", "анализ смысла",
                "построение онтологии", "извлечение знаний", "семантический разбор",
                "рассуждение по правилам", "категоризация понятий", "оценка достоверности",
                "поиск противоречия",
            ],
            "qualifiers": [
                "семантический", "логический", "онтологический", "концептуальный",
                "экспертный", "декларативный", "процедурный", "контекстный",
            ],
        },
        "en": {
            "subjects": [
                "knowledge", "ontology", "logic", "semantics", "concept", "meaning",
                "context", "reasoning", "knowledge base", "knowledge representation",
                "expert system", "hypothesis", "inference rule", "logical consequence",
                "thesaurus", "categorization", "definition", "argumentation",
                "contradiction", "semantic network", "knowledge extraction",
                "knowledge model", "predicate logic", "knowledge graph",
            ],
            "actions": [
                "logical inference", "hypothesis check", "meaning analysis",
                "ontology construction", "knowledge extraction", "semantic parsing",
                "rule reasoning", "concept categorization", "reliability check",
                "contradiction search",
            ],
            "qualifiers": [
                "semantic", "logical", "ontological", "conceptual",
                "expert", "declarative", "procedural", "contextual",
            ],
        },
    },
    "FLOW": {
        "ru": {
            "subjects": [
                "поток", "конвейер", "очередь", "планировщик", "пропускная способность",
                "буферизация", "конкурентность", "буфер", "этап обработки", "каскад",
                "маршрут данных", "цикл обработки", "последовательность этапов",
                "распределение нагрузки", "асинхронная задача", "потоковая обработка",
                "повторная обработка", "запуск конвейера", "остановка конвейера",
                "приоритет задачи", "время ожидания", "переполнение очереди",
                "обратное давление", "передача данных",
            ],
            "actions": [
                "буферизация данных", "планирование задач", "распределение нагрузки",
                "синхронизация этапов", "передача по конвейеру", "обработка очереди",
                "повторный запуск", "приоритизация задач", "балансировка потока",
                "маршрутизация данных",
            ],
            "qualifiers": [
                "потоковый", "асинхронный", "параллельный", "буферизованный",
                "каскадный", "поточный", "конвейерный", "циклический",
            ],
        },
        "en": {
            "subjects": [
                "stream", "pipeline", "queue", "scheduler", "throughput",
                "buffering", "concurrency", "buffer", "processing stage", "cascade",
                "data route", "processing cycle", "stage sequence", "load distribution",
                "asynchronous task", "stream processing", "reprocessing", "pipeline start",
                "pipeline stop", "task priority", "waiting time", "queue overflow",
                "backpressure", "data transfer",
            ],
            "actions": [
                "data buffering", "task scheduling", "load distribution",
                "stage synchronization", "pipeline transfer", "queue processing",
                "repeated launch", "task prioritization", "flow balancing",
                "data routing",
            ],
            "qualifiers": [
                "streaming", "asynchronous", "parallel", "buffered",
                "cascading", "concurrent", "pipelined", "cyclic",
            ],
        },
    },
    "TEXT": {
        "ru": {
            "subjects": [
                "текст", "слово", "предложение", "морфема", "синтаксис", "грамматика",
                "корпус", "токен", "лексема", "фраза", "абзац", "документ", "перевод",
                "орфография", "пунктуация", "разметка текста", "частеречная разметка",
                "лемматизация", "стемминг", "транслитерация", "стилометрия",
                "фразеология", "падеж", "алфавит",
            ],
            "actions": [
                "морфологический разбор", "синтаксический анализ", "токенизация текста",
                "лемматизация слов", "проверка орфографии", "расстановка пунктуации",
                "перевод предложений", "частеречная разметка", "разбор падежей",
                "транслитерация текста",
            ],
            "qualifiers": [
                "морфологический", "синтаксический", "лексический", "корпусный",
                "орфографический", "пунктуационный", "грамматический", "фразеологический",
            ],
        },
        "en": {
            "subjects": [
                "text", "word", "sentence", "morpheme", "syntax", "grammar",
                "corpus", "token", "lexeme", "phrase", "paragraph", "document",
                "translation", "spelling", "punctuation", "text tagging",
                "part-of-speech tagging", "lemmatization", "stemming", "transliteration",
                "stylometry", "phraseology", "case", "alphabet",
            ],
            "actions": [
                "morphological analysis", "syntax analysis", "text tokenization",
                "word lemmatization", "spelling check", "punctuation placement",
                "sentence translation", "part-of-speech tagging", "case parsing",
                "text transliteration",
            ],
            "qualifiers": [
                "morphological", "syntactic", "lexical", "corpus",
                "orthographic", "punctuation", "grammatical", "phraseological",
            ],
        },
    },
    "NUM": {
        "ru": {
            "subjects": [
                "число", "матрица", "вектор", "интеграл", "производная", "вероятность",
                "статистика", "оптимизация", "градиент", "спектр", "численный метод",
                "собственный вектор", "разложение матрицы", "ряд Тейлора", "предел",
                "комбинаторика", "распределение", "квантиль", "аппроксимация",
                "погрешность", "сходимость", "дискретная математика", "линейная алгебра",
                "математическое ожидание",
            ],
            "actions": [
                "численное интегрирование", "матричное разложение", "поиск градиента",
                "оценка вероятности", "статистическая проверка", "решение уравнения",
                "аппроксимация функции", "анализ сходимости", "вычисление спектра",
                "оценка погрешности",
            ],
            "qualifiers": [
                "численный", "матричный", "вероятностный", "статистический",
                "линейный", "дискретный", "непрерывный", "спектральный",
            ],
        },
        "en": {
            "subjects": [
                "number", "matrix", "vector", "integral", "derivative", "probability",
                "statistics", "optimization", "gradient", "spectrum", "numerical method",
                "eigenvector", "matrix decomposition", "Taylor series", "limit",
                "combinatorics", "distribution", "quantile", "approximation",
                "error estimate", "convergence", "discrete mathematics", "linear algebra",
                "mathematical expectation",
            ],
            "actions": [
                "numerical integration", "matrix decomposition", "gradient search",
                "probability estimate", "statistical test", "equation solving",
                "function approximation", "convergence analysis", "spectrum computation",
                "error estimate",
            ],
            "qualifiers": [
                "numerical", "matrix", "probabilistic", "statistical",
                "linear", "discrete", "continuous", "spectral",
            ],
        },
    },
    "COMPUTE": {
        "ru": {
            "subjects": [
                "компилятор", "байткод", "инструкция", "регистр", "загрузчик", "ассемблер",
                "ядро", "кэш", "драйвер", "сборка", "линковка", "объектный файл",
                "таблица символов", "виртуальная машина", "отладчик", "точка останова",
                "статический анализ", "генерация кода", "оптимизация компилятора",
                "машинный код", "прерывание", "контейнер", "изоляция процессов",
                "системный вызов",
            ],
            "actions": [
                "компиляция кода", "сборка модуля", "линковка библиотек",
                "генерация кода", "оптимизация компилятора", "статический анализ кода",
                "загрузка байткода", "распределение регистров", "отладка программы",
                "проверка таблицы символов",
            ],
            "qualifiers": [
                "компилируемый", "бинарный", "системный", "виртуальный",
                "кэшируемый", "регистровый", "аппаратный", "программный",
            ],
        },
        "en": {
            "subjects": [
                "compiler", "bytecode", "instruction", "register", "loader", "assembler",
                "kernel", "cache", "driver", "build", "linking", "object file",
                "symbol table", "virtual machine", "debugger", "breakpoint",
                "static analysis", "code generation", "compiler optimization",
                "machine code", "interrupt", "container", "process isolation",
                "system call",
            ],
            "actions": [
                "code compilation", "module build", "library linking",
                "code generation", "compiler optimization", "static code analysis",
                "bytecode loading", "register allocation", "program debugging",
                "symbol table check",
            ],
            "qualifiers": [
                "compiled", "binary", "system", "virtual",
                "cached", "register", "hardware", "software",
            ],
        },
    },
    "MIND": {
        "ru": {
            "subjects": [
                "мозг", "нейрон", "синапс", "эмоция", "внимание", "восприятие",
                "сознание", "познание", "поведение", "кора мозга", "когнитивная нагрузка",
                "рабочая память", "нейропластичность", "мотивация", "вознаграждение",
                "сенсорная интеграция", "принятие решений", "когнитивное искажение",
                "интуиция", "инсайт", "темперамент", "личность", "стресс", "адаптация",
            ],
            "actions": [
                "восприятие сигнала", "принятие решения", "когнитивная обработка",
                "обучение поведению", "эмоциональная реакция", "распределение внимания",
                "сенсорная интеграция", "формирование памяти", "оценка вознаграждения",
                "адаптация к стрессу",
            ],
            "qualifiers": [
                "когнитивный", "нейронный", "эмоциональный", "психический",
                "поведенческий", "сенсорный", "осознанный", "ментальный",
            ],
        },
        "en": {
            "subjects": [
                "brain", "neuron", "synapse", "emotion", "attention", "perception",
                "consciousness", "cognition", "behavior", "cortex", "cognitive load",
                "working memory", "neuroplasticity", "motivation", "reward",
                "sensory integration", "decision making", "cognitive bias",
                "intuition", "insight", "temperament", "personality", "stress",
                "adaptation",
            ],
            "actions": [
                "signal perception", "decision making", "cognitive processing",
                "behavioral learning", "emotional reaction", "attention allocation",
                "sensory integration", "memory formation", "reward evaluation",
                "stress adaptation",
            ],
            "qualifiers": [
                "cognitive", "neural", "emotional", "mental",
                "behavioral", "sensory", "conscious", "psychological",
            ],
        },
    },
    "NET": {
        "ru": {
            "subjects": [
                "пакет", "протокол", "маршрутизатор", "сетевой экран", "прокси", "шлюз",
                "сокет", "порт", "трафик", "соединение", "сессия", "домен", "адрес",
                "коммутатор", "канал связи", "проверка пакетов", "таблица маршрутов",
                "контрольная сумма", "беспроводной канал", "сертификат", "туннель",
                "балансировщик", "шифрование трафика", "фильтрация пакетов",
            ],
            "actions": [
                "передача пакетов", "маршрутизация трафика", "проверка контрольной суммы",
                "установка соединения", "фильтрация пакетов", "шифрование трафика",
                "разрешение адреса", "балансировка соединений", "работа прокси",
                "настройка шлюза",
            ],
            "qualifiers": [
                "сетевой", "пакетный", "протокольный", "маршрутный",
                "защищённый", "беспроводной", "проводной", "туннельный",
            ],
        },
        "en": {
            "subjects": [
                "packet", "protocol", "router", "firewall", "proxy", "gateway",
                "socket", "port", "traffic", "connection", "session", "domain",
                "address", "switch", "link channel", "packet check", "routing table",
                "checksum", "wireless channel", "certificate", "tunnel",
                "load balancer", "traffic encryption", "packet filtering",
            ],
            "actions": [
                "packet transfer", "traffic routing", "checksum check",
                "connection setup", "packet filtering", "traffic encryption",
                "address resolution", "connection balancing", "proxy operation",
                "gateway configuration",
            ],
            "qualifiers": [
                "network", "packet", "protocol", "routing",
                "secure", "wireless", "wired", "tunnel",
            ],
        },
    },
    "ACTION": {
        "ru": {
            "subjects": [
                "удаление", "фильтрация", "слияние", "вставка", "конвертация", "сжатие",
                "резервное копирование", "миграция", "сортировка", "копирование",
                "агрегация", "поиск", "преобразование", "рефакторинг", "очистка",
                "перезапуск", "развёртывание", "генерация отчёта", "пакетная обработка",
                "проверка целостности", "распаковка", "восстановление", "объединение",
                "разбиение",
            ],
            "actions": [
                "удаление дубликатов", "фильтрация записей", "слияние списков",
                "вставка элемента", "конвертация формата", "сжатие архива",
                "резервное копирование", "миграция схемы", "сортировка данных",
                "агрегация метрик",
            ],
            "qualifiers": [
                "пакетный", "групповой", "автоматический", "поэтапный",
                "массовый", "безопасный", "обратимый", "повторяемый",
            ],
        },
        "en": {
            "subjects": [
                "deletion", "filtering", "merging", "insertion", "conversion",
                "compression", "backup", "migration", "sorting", "copying",
                "aggregation", "search", "transformation", "refactoring", "cleanup",
                "restart", "deployment", "report generation", "batch processing",
                "integrity check", "extraction", "restoration", "union", "partitioning",
            ],
            "actions": [
                "duplicate deletion", "record filtering", "list merging",
                "element insertion", "format conversion", "archive compression",
                "backup creation", "schema migration", "data sorting",
                "metric aggregation",
            ],
            "qualifiers": [
                "batch", "group", "automatic", "staged",
                "mass", "safe", "reversible", "repeatable",
            ],
        },
    },
}

NEUTRAL = {
    "ru": ["в примере", "в обзоре", "в основе", "в методе", "в подходе", "в случае", "в проекте", "на практике"],
    "en": ["in an example", "in an overview", "in basics", "in a method", "in an approach", "in a case", "in a project", "in practice"],
}

TEMPLATES = {
    "ru": [
        "{s} и {t}",
        "{s} и {a}",
        "{a} и {t}",
        "{q} {s}",
        "{s} {n}",
        "{s} {n}",
        "{s}: {a}",
        "{a} и {s}",
    ],
    "en": [
        "{s} and {t}",
        "{s} and {a}",
        "{a} and {t}",
        "{q} {s}",
        "{s} {n}",
        "{s} {n}",
        "{s}: {a}",
        "{a} and {s}",
    ],
}

# Hard pairs: primary topic is the head of the phrase, secondary is the context.
# The validator additionally requires primary support >= secondary support.
HARD_SPECS: list[dict[str, Any]] = [
    {
        "pair": "GRAPH/NET", "primary": "GRAPH", "secondary": "NET",
        "ru_ops": ["остовное дерево", "обход графа", "матрица смежности",
                   "разрез графа", "раскраска графа", "компоненты связности",
                   "топологическая сортировка", "связность графа"],
        "ru_ctx": ["маршрутизаторов", "сетевых экранов", "прокси", "пакетов", "шлюзов"],
        "ru_templates": ["{op} для {ctx}"],
        "en_ops": ["spanning tree", "graph traversal", "adjacency matrix",
                   "edge cut", "graph coloring", "connected components",
                   "topological sort", "graph connectivity"],
        "en_ctx": ["routers", "firewalls", "proxies", "packets", "gateways"],
        "en_templates": ["{op} for {ctx}"],
    },
    {
        "pair": "GRAPH/NET", "primary": "NET", "secondary": "GRAPH",
        "ru_ops": ["маршрутизатор", "сетевой экран", "прокси", "шлюз",
                   "проверка пакетов", "туннель", "балансировщик", "фильтрация пакетов"],
        "ru_ctx": ["графе связей", "графе узлов", "графе вершин", "графе рёбер", "топологии"],
        "ru_templates": ["{op} в {ctx}"],
        "en_ops": ["router", "firewall", "proxy", "gateway",
                   "packet check", "tunnel", "load balancer", "packet filtering"],
        "en_ctx": ["a graph of links", "nodes", "vertices", "edges", "topology"],
        "en_templates": ["{op} over {ctx}", "{op} in {ctx}"],
    },
    {
        "pair": "COMPUTE/ACTION", "primary": "COMPUTE", "secondary": "ACTION",
        "ru_ops": ["сборка модуля", "компиляция кода", "кэш сборки", "генерация кода",
                   "линковка библиотек", "байткод модуля", "таблица символов",
                   "статический анализ кода"],
        "ru_ctx": ["рефакторинга", "преобразования кода", "поиска и замены",
                   "очистки кэша", "миграции"],
        "ru_templates": ["{op} после {ctx}"],
        "en_ops": ["module build", "code compilation", "build cache", "code generation",
                   "library linking", "module bytecode", "symbol table",
                   "static code analysis"],
        "en_ctx": ["refactoring", "code transformation", "search and replace",
                   "cache cleanup", "schema migration"],
        "en_templates": ["{op} after {ctx}"],
    },
    {
        "pair": "COMPUTE/ACTION", "primary": "ACTION", "secondary": "COMPUTE",
        "ru_ops": ["рефакторинг", "преобразование", "поиск и замена", "очистка",
                   "миграция", "перезапуск", "сжатие", "восстановление"],
        "ru_ctx": ["исходного модуля", "объектного файла", "кэша сборки",
                   "байткода", "таблицы символов"],
        "ru_templates": ["{op} {ctx}"],
        "en_ops": ["refactoring", "transformation", "search and replace", "cleanup",
                   "migration", "restart", "compression", "restoration"],
        "en_ctx": ["of the source module", "of the object file", "in the build cache",
                   "of the bytecode", "of the symbol table"],
        "en_templates": ["{op} {ctx}"],
    },
    {
        "pair": "KNOW/TEXT", "primary": "KNOW", "secondary": "TEXT",
        "ru_ops": ["извлечение знаний", "представление знаний", "онтология понятий",
                   "логический вывод", "семантические отношения", "экспертное правило",
                   "рассуждение", "модель знаний"],
        "ru_ctx": ["текста", "текстового корпуса", "предложений", "документов", "грамматики"],
        "ru_templates": ["{op} из {ctx}"],
        "en_ops": ["knowledge extraction", "knowledge representation", "concept ontology",
                   "logical inference", "semantic relations", "expert rule",
                   "reasoning", "knowledge model"],
        "en_ctx": ["text", "a text corpus", "sentences", "documents", "grammar"],
        "en_templates": ["{op} from {ctx}"],
    },
    {
        "pair": "KNOW/TEXT", "primary": "TEXT", "secondary": "KNOW",
        "ru_ops": ["грамматический разбор", "морфологический анализ",
                   "синтаксическое дерево", "токенизация текста",
                   "частеречная разметка", "пунктуация", "лемматизация", "перевод предложений"],
        "ru_ctx": ["семантическая разметка", "база знаний", "извлечение знаний",
                   "онтология", "контекст"],
        "ru_templates": ["{op} и {ctx}"],
        "en_ops": ["grammar parsing", "morphological analysis", "syntax tree",
                   "text tokenization", "part-of-speech tagging", "punctuation",
                   "lemmatization", "sentence translation"],
        "en_ctx": ["semantic tagging", "a knowledge base", "knowledge extraction",
                   "ontology", "context"],
        "en_templates": ["{op} and {ctx}"],
    },
    {
        "pair": "FLOW/NET", "primary": "FLOW", "secondary": "NET",
        "ru_ops": ["пропускная способность", "буферизация", "очередь",
                   "конвейер обработки", "распределение нагрузки", "поток событий",
                   "обратное давление", "планировщик задач"],
        "ru_ctx": ["сетевых пакетов", "маршрутизаторов", "сетевых экранов", "шлюзов"],
        "ru_templates": ["{op} для {ctx}"],
        "en_ops": ["throughput", "buffering", "queue", "processing pipeline",
                   "load distribution", "event stream", "backpressure", "task scheduler"],
        "en_ctx": ["network packets", "routers", "firewalls", "gateways"],
        "en_templates": ["{op} for {ctx}"],
    },
    {
        "pair": "FLOW/NET", "primary": "NET", "secondary": "FLOW",
        "ru_ops": ["маршрутизатор", "сетевой экран", "прокси", "шлюз",
                   "проверка пакетов", "туннель", "балансировщик", "фильтрация пакетов"],
        "ru_ctx": ["потоке данных", "потоковой обработке", "очереди", "конвейере", "буфере"],
        "ru_templates": ["{op} в {ctx}"],
        "en_ops": ["router", "firewall", "proxy", "gateway",
                   "packet check", "tunnel", "load balancer", "packet filtering"],
        "en_ctx": ["data flow", "stream processing", "a queue", "a pipeline", "a buffer"],
        "en_templates": ["{op} in {ctx}"],
    },
    {
        "pair": "MIND/NET", "primary": "MIND", "secondary": "NET",
        "ru_ops": ["внимание", "эмоция", "восприятие", "сознание", "познание",
                   "поведение", "когнитивная нагрузка", "принятие решений"],
        "ru_ctx": ["нейронная сеть", "сеть нейронов", "сетевые связи", "синапсы"],
        "ru_templates": ["{op} и {ctx}"],
        "en_ops": ["attention", "emotion", "perception", "consciousness", "cognition",
                   "behavior", "cognitive load", "decision making"],
        "en_ctx": ["neural network", "network of neurons", "network connections", "synapses"],
        "en_templates": ["{op} and {ctx}"],
    },
    {
        "pair": "NUM/GRAPH", "primary": "NUM", "secondary": "GRAPH",
        "ru_ops": ["матричный спектр", "спектральный анализ", "статистическая метрика",
                   "численная оценка", "матричное разложение", "алгебраический спектр",
                   "собственный вектор", "численный спектр"],
        "ru_ctx": ["графа", "матрицы смежности", "вершин графа", "теории графов"],
        "ru_templates": ["{op} {ctx}"],
        "en_ops": ["matrix spectrum", "spectral analysis", "statistical metric",
                   "numerical estimate", "matrix decomposition", "algebraic spectrum",
                   "eigenvector", "numerical spectrum"],
        "en_ctx": ["of a graph", "of an adjacency matrix", "of graph vertices",
                   "in graph theory"],
        "en_templates": ["{op} {ctx}"],
    },
]

TOKEN = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ_]+")
WORD = re.compile(r"[^\W]+(?:'[^\W]+)*", re.UNICODE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_existing() -> set[str]:
    existing: set[str] = set()
    for name in ("text_clusters.json", "text_clusters_manual.json",
                 "text_clusters_manual_v2.json"):
        path = ROOT / "training" / "corpora" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing.update(normalize(sample["query"]) for sample in payload["samples"])
    return existing


def make_analyzer():
    from training.ae_core import load_core
    trie = load_core().AEEnginePipeline(2000, 6).trie()
    zero = [-1.0] * 16

    def analyze(query: str) -> list[int]:
        _, _, counts = trie.compile_morpheme_analysis(query, zero)
        return [int(counts[CLUSTER_INDEX[name.lower()]]) for name in LABELS]

    return analyze


def unambiguous_candidates(name: str, lang: str) -> list[tuple[str, str]]:
    lex = LEXICON[name][lang]
    subjects = lex["subjects"]
    actions = lex["actions"]
    qualifiers = lex["qualifiers"]
    neutral = NEUTRAL[lang]
    templates = TEMPLATES[lang]
    out: list[tuple[str, str]] = []
    for index, subject in enumerate(subjects):
        if lang == "en":
            for q_index, qualifier in enumerate(qualifiers):
                out.append((templates[3].format(s=subject, q=qualifier),
                            f"{name}-{lang}-q{q_index}-{index}"))
        for n_index, word in enumerate(neutral):
            out.append((templates[4].format(s=subject, n=word),
                        f"{name}-{lang}-n{n_index}-{index}"))
        for a_index, action in enumerate(actions):
            out.append((templates[1].format(s=subject, a=action),
                        f"{name}-{lang}-a{a_index}-{index}"))
            out.append((templates[6].format(s=subject, a=action),
                        f"{name}-{lang}-c{a_index}-{index}"))
            out.append((templates[7].format(s=subject, a=action),
                        f"{name}-{lang}-i{a_index}-{index}"))
        for other_index, other in enumerate(subjects):
            if other_index == index:
                continue
            out.append((templates[0].format(s=subject, t=other),
                        f"{name}-{lang}-t{index}-{other_index}"))
            out.append((templates[2].format(a=subject, t=other),
                        f"{name}-{lang}-r{index}-{other_index}"))
    return out


def hard_candidates(spec: dict[str, Any]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for lang in ("ru", "en"):
        ops = spec[f"{lang}_ops"]
        ctx = spec[f"{lang}_ctx"]
        templates = spec[f"{lang}_templates"]
        for op_index, operation in enumerate(ops):
            for c_index, context in enumerate(ctx):
                for t_index, template in enumerate(templates):
                    query = template.format(op=operation, ctx=context)
                    family = (f"{spec['primary']}-hard-{spec['pair'].replace('/', '-')}"
                              f"-{lang}-o{op_index}-c{c_index}")
                    out.append((query, family, spec["pair"]))
    return out


def digest(family: str) -> int:
    return int(hashlib.sha1(family.encode("utf-8")).hexdigest()[:8], 16)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "training/corpora/text_clusters_synthetic.json")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--no-core-check", action="store_true",
                        help="skip trie validation (not recommended)")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    existing = load_existing()
    analyze = None if args.no_core_check else make_analyzer()

    # Hard candidates first: they are rarer and must not crowd out pure topics.
    hard_by_primary: dict[str, list[tuple[str, str, str]]] = {name: [] for name in LABELS}
    seen_hard: set[str] = set()
    for spec in HARD_SPECS:
        items = hard_candidates(spec)
        rng.shuffle(items)
        for query, family, pair in items:
            key = normalize(query)
            if key in existing or key in seen_hard:
                continue
            if analyze is not None:
                counts = analyze(query)
                primary = counts[LABELS.index(spec["primary"])]
                secondary = counts[LABELS.index(spec["secondary"])]
                if primary <= 0 or secondary <= 0 or primary < secondary:
                    continue
            hard_by_primary[spec["primary"]].append((query, family, pair))
            seen_hard.add(key)

    samples: list[dict[str, Any]] = []
    for label, name in enumerate(LABELS):
        lang_candidates: dict[str, list[tuple[str, str]]] = {}
        for lang in ("ru", "en"):
            items = unambiguous_candidates(name, lang)
            rng.shuffle(items)
            lang_candidates[lang] = items

        selected: list[tuple[str, str, str | None, str]] = []
        seen_selected: set[str] = set()

        # Up to 10% of each class may be hard pairs (25 per language).
        hard_cap_per_language = max(1, args.per_class // 20)
        for lang in ("ru", "en"):
            hard = list(hard_by_primary[name])
            rng.shuffle(hard)
            chosen = sum(1 for item in selected if item[3] == lang)
            for query, family, pair in hard:
                if chosen >= hard_cap_per_language:
                    break
                is_ru = bool(re.search(r"[А-Яа-яЁё]", query))
                if (lang == "ru") != is_ru:
                    continue
                key = normalize(query)
                if key in seen_selected or key in existing:
                    continue
                selected.append((query, family, pair, lang))
                seen_selected.add(key)
                chosen += 1

        # Pure single-topic candidates fill the rest.  Clearly dominant mixed
        # candidates are kept as a fallback if the lexicon runs out.
        dominant: dict[str, list[tuple[str, str]]] = {"ru": [], "en": []}
        for lang in ("ru", "en"):
            need = args.per_class // 2 - sum(1 for item in selected if item[3] == lang)
            for query, family in lang_candidates[lang]:
                if need <= 0:
                    break
                key = normalize(query)
                if key in seen_selected or key in existing:
                    continue
                if analyze is not None:
                    counts = analyze(query)
                    intended = counts[label]
                    if intended <= 0:
                        continue
                    others = [counts[i] for i in range(len(LABELS))
                              if i != label and counts[i] > 0]
                    if others:
                        if intended >= 2 * max(others) and intended >= 2:
                            dominant[lang].append((query, family))
                        continue
                selected.append((query, family, None, lang))
                seen_selected.add(key)
                need -= 1

        for lang in ("ru", "en"):
            need = args.per_class // 2 - sum(1 for item in selected if item[3] == lang)
            for query, family in dominant[lang]:
                if need <= 0:
                    break
                key = normalize(query)
                if key in seen_selected or key in existing:
                    continue
                selected.append((query, family, "dominant", lang))
                seen_selected.add(key)
                need -= 1

        if len(selected) != args.per_class:
            raise SystemExit(
                f"{name}: only {len(selected)} candidates, need {args.per_class}; "
                "expand the lexicon or relax --per-class"
            )
        rng.shuffle(selected)
        for query, family, pair, lang in selected:
            sample = {
                "query": query.strip(),
                "label": label,
                "cluster": name,
                "split": "train",
                "source": "synthetic-v1",
                "family": family,
                "language": "ru" if lang == "ru" else "en",
            }
            if pair:
                sample["notes"] = f"hard {pair}: primary {name} dominates the context"
            samples.append(sample)

    payload = {
        "name": "text_clusters_synthetic",
        "version": 1,
        "labels": LABELS,
        "samples": samples,
        "meta": {
            "generator": "tools/make_text_cluster_synthetic.py",
            "seed": args.seed,
            "per_class": args.per_class,
            "samples": len(samples),
            "split": "train",
            "core_validated": not args.no_core_check,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {len(samples)} samples to {args.output}")
    for name in LABELS:
        rows = [s for s in samples if s["cluster"] == name]
        ru = sum(1 for s in rows if s["language"] == "ru")
        hard = sum(1 for s in rows if "notes" in s)
        print(f"  {name:8s} {len(rows):4d}  ru={ru:3d}  en={len(rows)-ru:3d}  hard={hard:3d}")


if __name__ == "__main__":
    main()
