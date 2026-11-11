"""Classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores, confusion matrix."""
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    top2 = np.argsort(probs, axis=1)[:, -2:]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top2_accuracy": float(np.mean([t in row for t, row in zip(y_true, top2)])),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
