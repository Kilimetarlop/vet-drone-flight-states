"""Five-class names and metrics shared by the branches.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


CLASS_NAMES = (
    "LINEAR_CRUISE",
    "CURVED_TURN",
    "HOVER",
    "TAKEOFF_ASCEND",
    "DESCEND_LAND",
)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        prediction,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(target, prediction, labels=labels)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }
