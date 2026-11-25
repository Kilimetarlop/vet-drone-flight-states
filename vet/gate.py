"""Confidence-gated correction, the published-rule baseline of the paper (Section V-B).

It uses VET's candidates and per-fold configuration, but replaces the direction test by a threshold gamma on the
largest candidate probability: a change is admitted when max_k z_ik >= gamma. Gamma is chosen on a 0.05 grid from the
held-out source predictions by the accuracy and Macro-F1 conditions of eq. (12) and the ranking of eq. (13), ties
broken toward a higher threshold.
"""

from __future__ import annotations

import numpy as np

from .candidate import candidate_posterior
from .metrics import ranking_key, scores

THRESHOLDS = tuple(np.round(np.arange(0.40, 0.901, 0.05), 2))


def gate_reports(z: np.ndarray, h: np.ndarray, gamma: float) -> np.ndarray:
    return np.where(z.max(1) >= gamma, z.argmax(1), np.asarray(h, dtype=np.int64))


def select_threshold(y, product, d, h, s, lam, eta, thresholds=THRESHOLDS) -> float | None:
    """Threshold of the gate from held-out source predictions; None when no threshold is eligible."""
    z = candidate_posterior(d, h, s, lam, eta)
    base = scores(y, h)
    products = sorted(set(np.asarray(product).tolist()))
    base_f1 = {g: scores(y[product == g], h[product == g])["macro_f1"] for g in products}
    best = None
    for gamma in thresholds:
        pred = gate_reports(z, h, gamma)
        m = scores(y, pred)
        if not (m["accuracy"] >= base["accuracy"] - 1e-12 and m["macro_f1"] > base["macro_f1"] + 1e-12):
            continue
        delta = [scores(y[product == g], pred[product == g])["macro_f1"] - base_f1[g] for g in products]
        key = (sum(x >= -1e-12 for x in delta), float(np.mean(delta)), *ranking_key(m), gamma)
        if best is None or key > best[0]:
            best = (key, float(gamma))
    return None if best is None else best[1]


def run_confidence_gate(y, product, holdout) -> tuple[np.ndarray, list]:
    """Gate every outer fold of a `vet.model.HoldoutResult`, with VET's selected configuration of that fold."""
    y, product = np.asarray(y, dtype=np.int64), np.asarray(product)
    report = holdout.reference.copy()
    chosen = []
    for f in holdout.folds:
        if f.selection is None:
            chosen.append((f.target, None))
            continue
        src = f.source_rows
        gamma = select_threshold(y[src], product[src], f.source_d, f.source_h, f.source_scores,
                                 f.selection.lam, f.selection.eta)
        chosen.append((f.target, gamma))
        if gamma is not None:
            z = candidate_posterior(f.d, f.h, f.scores, f.selection.lam, f.selection.eta)
            report[f.target_rows] = gate_reports(z, f.h, gamma)
    return report, chosen
