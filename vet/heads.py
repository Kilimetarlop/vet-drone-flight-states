"""The three conditional heads of the chained state posterior (paper eq. (5), Section IV-C).

Each head maps one recording view to one conditional probability with the same low-capacity model: per-feature
standardization, the leading 16 principal directions, and a class-balanced L2 logistic regression (C = 1). Head j is
fitted on the source recordings F_j that its question compares, and scores every recording:

    hover    s = Pr(Hover | x)                         F_s: all source recordings          view: envelope
    vertical v = Pr(Ascend or Descend | moving, x)     F_v: source recordings not Hover    view: envelope
    ascent   u = Pr(Ascend | Ascend or Descend, x)     F_u: source Ascend and Descend      view: STFT
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .states import ASCEND, HOVER

HEADS = ("hover", "vertical", "ascent")
HEAD_VIEW = {"hover": "envelope", "vertical": "envelope", "ascent": "stft"}
SEED = 20260831
SEED_OFFSET = {"hover": 1, "vertical": 2, "ascent": 3}
COMPONENTS = 16


def fit_set(head: str, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows of F_j among `rows`, and the binary target of head j for every recording."""
    if head == "hover":
        return rows, (y == HOVER).astype(int)
    if head == "vertical":
        return rows[y[rows] != HOVER], (y >= ASCEND).astype(int)
    if head == "ascent":
        return rows[y[rows] >= ASCEND], (y == ASCEND).astype(int)
    raise KeyError(head)


def make_head(n_rows: int, n_features: int, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=min(COMPONENTS, n_features, n_rows - 1), svd_solver="full", random_state=seed)),
        ("logistic", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000,
                                        random_state=seed)),
    ])


def fit_heads(views: dict[str, np.ndarray], y: np.ndarray, train: np.ndarray, seeds: dict[str, int]) -> dict:
    """Fit the three heads on the training recordings `train` (indices into the view arrays)."""
    models = {}
    for head in HEADS:
        x = views[HEAD_VIEW[head]]
        rows, target = fit_set(head, y, np.asarray(train))
        models[head] = make_head(len(rows), x.shape[1], seeds[head]).fit(x[rows], target[rows])
    return models


def score_heads(models: dict, views: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Scores [s, v, u] of the recordings `rows`, shape (len(rows), 3)."""
    return np.column_stack([models[h].predict_proba(views[HEAD_VIEW[h]][rows])[:, 1] for h in HEADS])


def outer_seeds(fold: int) -> dict[str, int]:
    """Seed schedule of the paper for heads fitted on all source products of outer fold `fold`."""
    return {h: SEED + 100 * fold + SEED_OFFSET[h] for h in HEADS}


def inner_seeds(fold: int, inner: int) -> dict[str, int]:
    """Seed schedule of the paper for heads fitted without the `inner`-th source product of outer fold `fold`."""
    return {h: SEED + 1000 * fold + 10 * inner + SEED_OFFSET[h] for h in HEADS}
