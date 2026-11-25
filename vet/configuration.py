"""Source-only configuration (paper eqs. (12) and (13)).

For each configuration theta = (lambda, eta), the candidate of every held-out source prediction is computed, the
direction test learns the admitted set on those predictions, and eq. (11) gives source reports. A configuration is
eligible when it keeps the source accuracy, raises the source Macro-F1, and reports every state; among eligible ones the
lexicographic key prefers gains on more source products before gains in pooled scores, then a smaller lambda and a
larger eta. Everything here reads held-out source predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidate import CONFIGURATIONS, candidate_posterior
from .direction_test import R_MIN, SHARE, admitted_directions, decide
from .metrics import ranking_key, scores


@dataclass
class Selection:
    lam: float
    eta: float
    admitted: np.ndarray                       # (4, 4) boolean, eq. (10)
    key: tuple = field(repr=False, default=())


def source_reports(y, d, h, s, lam, eta, r_min=R_MIN, share=SHARE, admitted=None):
    """Candidates, admitted set, and eq. (11) reports on held-out source predictions under one configuration."""
    c = candidate_posterior(d, h, s, lam, eta).argmax(1)
    if admitted is None:
        admitted = admitted_directions(h, c, y, r_min, share)
    return c, admitted, decide(h, c, admitted)


def select_configuration(y: np.ndarray, product: np.ndarray, d: np.ndarray, h: np.ndarray, s: np.ndarray,
                         configurations=CONFIGURATIONS, r_min: int = R_MIN,
                         share: float = SHARE) -> Selection | None:
    """Eqs. (12)-(13) on the held-out source predictions M_t of one fold.

    y, product: labels and product names of the held-out source predictions; d, h: their reference posterior and judged
    state; s: their head scores [s, v, u]. Returns None when no configuration is eligible (the fold keeps the reference).
    """
    base = scores(y, h)
    products = sorted(set(np.asarray(product).tolist()))
    base_f1 = {g: scores(y[product == g], h[product == g])["macro_f1"] for g in products}
    best = None
    for lam, eta in configurations:
        _, admitted, pred = source_reports(y, d, h, s, lam, eta, r_min, share)
        m = scores(y, pred)
        eligible = (m["accuracy"] >= base["accuracy"] - 1e-12 and m["macro_f1"] > base["macro_f1"] + 1e-12
                    and bool(np.all(np.bincount(pred, minlength=4) > 0)))
        if not eligible:
            continue
        delta = [scores(y[product == g], pred[product == g])["macro_f1"] - base_f1[g] for g in products]
        key = (sum(x >= -1e-12 for x in delta), float(np.mean(delta)), *ranking_key(m), -lam, eta)
        if best is None or key > best.key:
            best = Selection(lam, eta, admitted, key)
    return best
