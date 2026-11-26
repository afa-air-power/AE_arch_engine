"""Small, dependency-light GraphSAGE baseline for AE graph algorithm integration.

The model is intentionally implemented with NumPy first: this makes the data
contract and CPU reference deterministic on machines where ROCm/PyTorch is not
available.  The graph algorithms remain first-class features: Dinitz-like
features, MST bottlenecks, centroid hops, and path statistics are accepted as
edge/node features rather than discarded by the learned model.

ADR: docs/adr/0012-trainable-graph-contour.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ADR-0012: docs/adr/0012-trainable-graph-contour.md
@dataclass
class GraphDataset:
    node_features: np.ndarray
    adjacency: np.ndarray
    labels: np.ndarray
    split: np.ndarray

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "GraphDataset":
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        features = np.asarray(data["node_features"], dtype=np.float32)
        adjacency = np.asarray(data.get("adjacency", []), dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        split = np.asarray(data["split"], dtype=str)
        n = len(labels)
        if features.ndim != 2 or features.shape[0] != n:
            raise ValueError("node_features must have shape [num_nodes, num_features]")
        # Пустой adjacency — это граф без рёбер.  Для 'none' не нужно хранить
        # и умножать плотную матрицу n×n: на больших корпусах это сотни МБ.
        if adjacency.size and adjacency.shape != (n, n):
            raise ValueError("adjacency must have shape [num_nodes, num_nodes] or be empty")
        if split.shape != (n,) or not set(split).issubset({"train", "val", "test"}):
            raise ValueError("split must contain train, val, or test for every node")
        return cls(features, adjacency, labels, split)

    def normalized_adjacency(self) -> np.ndarray | None:
        if self.adjacency.size == 0 or not np.any(self.adjacency):
            return None
        a = self.adjacency.copy()
        np.fill_diagonal(a, 1.0)
        degree = a.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
        return inv_sqrt[:, None] * a * inv_sqrt[None, :]


# ADR-0012: docs/adr/0012-trainable-graph-contour.md
# Â = D^{-1/2}(A + I)D^{-1/2}; при пустом графе агрегация тождественна.
class GraphSAGEClassifier:
    """Two-layer mean-aggregation classifier with explicit backward pass."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.linear = hidden_dim <= 0
        if self.linear:
            self.hidden_dim = 0
            self.w1 = np.zeros((input_dim, 0), dtype=np.float32)
            self.b1 = np.zeros(0, dtype=np.float32)
            self.w2 = (rng.standard_normal((input_dim, classes)) * np.sqrt(1.0 / input_dim)).astype(np.float32)
        else:
            self.hidden_dim = hidden_dim
            scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
            self.w1 = (rng.standard_normal((input_dim, hidden_dim)) * scale1).astype(np.float32)
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.w2 = (rng.standard_normal((hidden_dim, classes)) * np.sqrt(2.0 / (hidden_dim + classes))).astype(np.float32)
        self.b2 = np.zeros(classes, dtype=np.float32)
        self.feature_mean = np.zeros(input_dim, dtype=np.float32)
        self.feature_std = np.ones(input_dim, dtype=np.float32)

    def fit_normalization(self, x: np.ndarray, indices: np.ndarray) -> None:
        subset = x[indices] if len(indices) else x
        self.feature_mean = subset.mean(axis=0).astype(np.float32)
        std = subset.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.feature_std = std

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    @staticmethod
    def _aggregate(normalized_adjacency: np.ndarray | None, values: np.ndarray) -> np.ndarray:
        # Пустой граф — это тождественная агрегация.  Так датасет без рёбер
        # не требует плотной матрицы n×n и не тратит на неё память/время.
        if normalized_adjacency is None:
            return values
        return normalized_adjacency @ values

    def forward(self, x: np.ndarray, normalized_adjacency: np.ndarray | None):
        x = self._normalize(x)
        if self.linear:
            h0 = self._aggregate(normalized_adjacency, x)
            logits = h0 @ self.w2 + self.b2
            return logits, (h0, None, None, h0)
        h0 = self._aggregate(normalized_adjacency, x)
        z1 = h0 @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        h2 = self._aggregate(normalized_adjacency, h1)
        logits = h2 @ self.w2 + self.b2
        return logits, (h0, z1, h1, h2)

    def loss_and_grad(self, x, a, labels, indices, weight_decay=0.0):
        logits, cache = self.forward(x, a)
        selected = logits[indices]
        shifted = selected - selected.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(np.maximum(probabilities[np.arange(len(indices)), labels[indices]], 1e-12)).mean()
        dlogits = np.zeros_like(logits)
        dlogits[indices] = probabilities
        dlogits[indices, labels[indices]] -= 1.0
        dlogits /= len(indices)
        h0, z1, h1, h2 = cache
        if self.linear:
            grad_w2 = h0.T @ dlogits + weight_decay * self.w2
            grad_b2 = dlogits.sum(axis=0)
            return float(loss), (None, None, grad_w2, grad_b2), logits
        grad_w2 = h2.T @ dlogits + weight_decay * self.w2
        grad_b2 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.w2.T
        dh1 = dh2 if a is None else a.T @ dh2
        dz1 = dh1 * (z1 > 0.0)
        grad_w1 = h0.T @ dz1 + weight_decay * self.w1
        grad_b1 = dz1.sum(axis=0)
        return float(loss), (grad_w1, grad_b1, grad_w2, grad_b2), logits

    def predict(self, x, a):
        return self.forward(x, a)[0].argmax(axis=1)

    def state_dict(self):
        return {
            "linear": self.linear,
            "hidden_dim": self.hidden_dim,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state):
        hidden_dim = 0 if state.get("linear") else int(state["hidden_dim"])
        model = cls(len(state["feature_mean"]), hidden_dim, len(state["w2"][0]))
        for key in ("w1", "b1", "w2", "b2", "feature_mean", "feature_std"):
            setattr(model, key, np.asarray(state[key], dtype=np.float32))
        return model


def _metrics(predictions, labels, indices):
    y_true, y_pred = labels[indices], predictions[indices]
    accuracy = float(np.mean(y_true == y_pred)) if len(indices) else 0.0
    f1_values = []
    for cls in np.unique(labels):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        if tp + fp + fn:
            f1_values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0}


# ADR-0012: docs/adr/0012-trainable-graph-contour.md (Adam, early stopping по F1)
def train(dataset: GraphDataset, epochs=100, learning_rate=1e-2, hidden_dim=32, seed=42, checkpoint=None, weight_decay=0.0, patience=0):
    classes = int(dataset.labels.max()) + 1
    model = GraphSAGEClassifier(dataset.node_features.shape[1], hidden_dim, classes, seed)
    a = dataset.normalized_adjacency()
    train_idx = np.flatnonzero(dataset.split == "train")
    val_idx = np.flatnonzero(dataset.split == "val")
    if len(train_idx) == 0:
        raise ValueError("dataset has no train split")
    model.fit_normalization(dataset.node_features, train_idx)

    # Adam: baseline was sensitive to feature scale, and the algorithm features
    # mix [0,1] ratios with raw counters before normalization.
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    moments = {key: np.zeros_like(getattr(model, key)) for key in ("w1", "b1", "w2", "b2")}
    velocities = {key: np.zeros_like(getattr(model, key)) for key in ("w1", "b1", "w2", "b2")}

    def validation_loss() -> float:
        if len(val_idx) == 0:
            return float("inf")
        value, _, _ = model.loss_and_grad(dataset.node_features, a, dataset.labels, val_idx, weight_decay)
        return value

    best = (float("inf"), None)
    best_val = float("inf")
    best_val_state = None
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        loss, grads, _ = model.loss_and_grad(dataset.node_features, a, dataset.labels, train_idx, weight_decay)
        for parameter, gradient in zip(("w1", "b1", "w2", "b2"), grads):
            if gradient is None:
                continue
            moments[parameter] = beta1 * moments[parameter] + (1 - beta1) * gradient
            velocities[parameter] = beta2 * velocities[parameter] + (1 - beta2) * gradient * gradient
            m_hat = moments[parameter] / (1 - beta1 ** epoch)
            v_hat = velocities[parameter] / (1 - beta2 ** epoch)
            setattr(model, parameter,
                    getattr(model, parameter) - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
        pred = model.predict(dataset.node_features, a)
        val_loss = validation_loss()
        val_metrics = _metrics(pred, dataset.labels, val_idx)
        record = {"epoch": epoch, "loss": loss, "val_loss": val_loss,
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        if loss < best[0]:
            best = (loss, model.state_dict())
        # Отбор по validation macro-F1: на малых корпусах validation loss
        # растёт почти сразу, а F1 ещё держит полезную эпоху.
        improved = val_metrics["macro_f1"] > best_f1 + 1e-9
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_val = val_loss
            best_val_state = model.state_dict()
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                history[-1]["early_stop"] = True
                break
    # Восстанавливаем состояние по validation macro-F1, а не по train loss:
    # иначе checkpoint фиксирует максимально переобученную эпоху.
    restore = best_val_state if best_val_state is not None else best[1]
    if restore is not None:
        model = GraphSAGEClassifier.from_state_dict(restore)
    if checkpoint:
        pathlib.Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(checkpoint).write_text(json.dumps({"model": model.state_dict(), "history": history}, indent=2), encoding="utf-8")
    return model, history


# ADR-0012: docs/adr/0012-trainable-graph-contour.md
def evaluate(model, dataset: GraphDataset):
    pred = model.predict(dataset.node_features, dataset.normalized_adjacency())
    return {name: _metrics(pred, dataset.labels, np.flatnonzero(dataset.split == name)) for name in ("train", "val", "test")}


# ADR-0012: docs/adr/0012-trainable-graph-contour.md
def stratified_folds(labels: np.ndarray, folds: int, seed: int = 42) -> np.ndarray:
    """Stratified fold assignment so every fold sees every class."""
    rng = np.random.default_rng(seed)
    assignment = np.zeros(len(labels), dtype=np.int64)
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignment[index] = position % folds
    return assignment


# ADR-0012: docs/adr/0012-trainable-graph-contour.md
def cross_validate(dataset: GraphDataset, folds: int = 5, epochs: int = 200,
                   learning_rate: float = 1e-2, hidden_dim: int = 0,
                   weight_decay: float = 1e-3, seed: int = 42) -> dict:
    """Stratified k-fold estimate: a single 18-sample test split is too noisy."""
    assignment = stratified_folds(dataset.labels, folds, seed)
    fold_metrics = []
    for fold in range(folds):
        split = np.where(assignment == fold, "test", "train").astype(object)
        train_indices = np.flatnonzero(split == "train")
        # Внутренний validation для early stopping, чтобы held-out fold
        # не участвовал в выборе эпохи.
        rng = np.random.default_rng(seed + fold)
        inner = rng.permutation(train_indices)
        val_size = max(1, len(inner) // 10)
        split[inner[:val_size]] = "val"
        fold_dataset = GraphDataset(dataset.node_features, dataset.adjacency, dataset.labels, split)
        model, _ = train(fold_dataset, epochs=epochs, learning_rate=learning_rate,
                         hidden_dim=hidden_dim, seed=seed + fold,
                         weight_decay=weight_decay, patience=0)
        metrics = evaluate(model, fold_dataset)["test"]
        fold_metrics.append(metrics)
    accuracies = np.asarray([m["accuracy"] for m in fold_metrics])
    f1_scores = np.asarray([m["macro_f1"] for m in fold_metrics])
    return {
        "folds": folds,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "macro_f1_mean": float(f1_scores.mean()),
        "macro_f1_std": float(f1_scores.std()),
        "per_fold": fold_metrics,
    }


def load_checkpoint(path):
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return GraphSAGEClassifier.from_state_dict(payload["model"])


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=0,
                        help="0 = linear (robust on small corpora), >0 = hidden GraphSAGE layer")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/ae_graph.json")
    parser.add_argument("--cv", type=int, default=0, help="stratified k-fold evaluation instead of the fixed split")
    args = parser.parse_args(argv)
    dataset = GraphDataset.load(args.dataset)
    if args.cv > 1:
        report = cross_validate(dataset, folds=args.cv, epochs=args.epochs,
                                learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
                                weight_decay=args.weight_decay)
        print(json.dumps({"cross_validation": report}, indent=2))
        return
    model, history = train(dataset, args.epochs, args.learning_rate, args.hidden_dim,
                           checkpoint=args.checkpoint, weight_decay=args.weight_decay,
                           patience=args.patience)
    print(json.dumps({"epochs": len(history), "metrics": evaluate(model, dataset)}, indent=2))


if __name__ == "__main__":
    main()
