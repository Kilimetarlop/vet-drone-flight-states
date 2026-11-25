"""Evaluation measures and statistics (paper Sections III-D and V-G)."""

from __future__ import annotations

import itertools
from math import comb, sqrt
from typing import Any

import numpy as np


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Accuracy, balanced accuracy (mean per-state recall), Macro-F1, per-state recall, and the confusion matrix."""
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.int64)), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(4), where=denom > 0)
    correct = int(tp.sum())
    return {
        "correct": correct,
        "total": int(len(y)),
        "accuracy": correct / len(y),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "recall": recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def ranking_key(m: dict[str, Any]) -> tuple[float, float, float]:
    """Pooled entries of eq. (13), in order: Macro-F1, balanced accuracy, accuracy."""
    return (m["macro_f1"], m["balanced_accuracy"], m["accuracy"])


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)


def paired(y: np.ndarray, base: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    """Repairs, damages, and neutral changes of `new` against `base`, with the exact McNemar p value."""
    changed = base != new
    repairs = int(np.sum(changed & (new == y) & (base != y)))
    damages = int(np.sum(changed & (base == y) & (new != y)))
    return {"changed": int(changed.sum()), "repairs": repairs, "damages": damages,
            "neutral": int(changed.sum()) - repairs - damages,
            "mcnemar_exact_p": exact_binomial_two_sided(repairs, repairs + damages)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip test of the summed per-product differences."""
    diffs = np.asarray(diffs, dtype=float)
    flips = np.array(list(itertools.product((-1, 1), repeat=len(diffs))))
    null = (flips * np.abs(diffs)).sum(1)
    return float(np.mean(np.abs(null) >= abs(diffs.sum())))


def product_bootstrap(y: np.ndarray, p: np.ndarray, base: np.ndarray, product: np.ndarray, n_boot: int = 10_000,
                      seed: int = 20260831, products: list[str] | None = None) -> dict[str, list[float]]:
    """95% intervals of the differences in accuracy, Macro-F1, and balanced accuracy under product resampling.

    Products are resampled in the order `products` (default: sorted), which fixes the stream of the generator.
    """
    rng = np.random.default_rng(seed)
    products = sorted(set(np.asarray(product).tolist())) if products is None else list(products)
    idx_of = {g: np.flatnonzero(product == g) for g in products}
    out = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(n_boot):
        pick = rng.choice(len(products), size=len(products), replace=True)
        idx = np.concatenate([idx_of[products[j]] for j in pick])
        m, mb = scores(y[idx], p[idx]), scores(y[idx], base[idx])
        for k in out:
            out[k].append(m[k] - mb[k])
    return {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in out.items()}


def auc(labels: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged)."""
    labels, score = np.asarray(labels), np.asarray(score, dtype=float)
    pos, neg = score[labels == 1], score[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate((pos, neg))
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    vals = allv[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
