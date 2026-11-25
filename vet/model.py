"""VET under whole-product holdout (paper Algorithm 1) and the deployed monitor.

The reference classifier is supplied by the caller. It is any callable

    reference(train, test) -> (h, d)

that fits on the recordings `train` (labels and side information are the caller's business) and returns, for the
recordings `test`, the judged state h (n,) in 0..3 and the four-state reference posterior d (n, 4). The paper's
reference is the authors' hierarchical classifier of Section IV-B; `PrecomputedReference` serves saved outputs of any
reference so that VET can be rerun without refitting it.

Splits follow eqs. (1) and (2): outer fold t fits on the source set S_t and reports the target product t; the held-out
source predictions M_t come from fitting on G minus {t, g} and predicting each source product g.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .candidate import CONFIGURATIONS, candidate_posterior
from .configuration import Selection, select_configuration, source_reports
from .direction_test import R_MIN, SHARE, decide
from .heads import fit_heads, inner_seeds, outer_seeds, score_heads

Reference = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class Fold:
    target: str
    selection: Selection | None                      # None: no eligible configuration, the reference is reported
    source_rows: np.ndarray                          # indices of S_t
    target_rows: np.ndarray                          # indices of T_t
    source_h: np.ndarray                             # held-out source predictions M_t (aligned with source_rows)
    source_d: np.ndarray
    source_scores: np.ndarray
    h: np.ndarray                                    # target judgments, posteriors, scores, candidates, reports
    d: np.ndarray
    scores: np.ndarray
    candidate: np.ndarray
    report: np.ndarray


@dataclass
class HoldoutResult:
    report: np.ndarray                               # VET report for every recording
    reference: np.ndarray                            # reference judgment h for every recording
    folds: list[Fold] = field(default_factory=list)


def heldout_source_predictions(y, product, views, reference: Reference, fold: int, source: np.ndarray,
                               score_fn=None):
    """M_t of eq. (2): reference outputs and head scores of every source product g, fitted on G minus {t, g}."""
    n = len(source)
    h, d, s = np.zeros(n, dtype=np.int64), np.zeros((n, 4)), np.zeros((n, 3))
    for i, g in enumerate(dict.fromkeys(product[source].tolist())):    # source products in data order
        val_mask = product[source] == g
        fit, val = source[~val_mask], source[val_mask]
        h[val_mask], d[val_mask] = reference(fit, val)
        if score_fn is not None:
            s[val_mask] = score_fn(fit, val, inner_seeds(fold, i))
        else:
            s[val_mask] = score_heads(fit_heads(views, y, fit, inner_seeds(fold, i)), views, val)
    return h, d, s


def run_whole_product_holdout(y: np.ndarray, product: np.ndarray, views: dict[str, np.ndarray] | None,
                              reference: Reference, *, configurations=CONFIGURATIONS, r_min: int = R_MIN,
                              share: float = SHARE, fixed: tuple[float, float] | None = None,
                              folds: list[str] | None = None, score_fn=None) -> HoldoutResult:
    """Algorithm 1: one outer fold per product t (sorted order by default).

    views: {'envelope': (n, 454), 'stft': (n, 1524)} recording views (vet.views); y: four-state labels (used only on
    source rows); product: product name of every recording. `fixed` holds the configuration instead of selecting it.
    `score_fn(train, test, seeds) -> (len(test), 3)` replaces the heads (for example, by saved scores).
    """
    y, product = np.asarray(y, dtype=np.int64), np.asarray(product)
    folds = sorted(set(product.tolist())) if folds is None else list(folds)
    report = np.full(len(y), -1, dtype=np.int64)
    reference_h = np.full(len(y), -1, dtype=np.int64)
    out = HoldoutResult(report, reference_h)
    for o, t in enumerate(folds):
        src, tgt = np.flatnonzero(product != t), np.flatnonzero(product == t)
        sh, sd, ss = heldout_source_predictions(y, product, views, reference, o, src, score_fn)   # lines 2-6
        if fixed is None:
            sel = select_configuration(y[src], product[src], sd, sh, ss, configurations, r_min, share)  # 7-13
        else:
            _, admitted, _ = source_reports(y[src], sd, sh, ss, *fixed, r_min=r_min, share=share)
            sel = Selection(fixed[0], fixed[1], admitted)
        h, d = reference(src, tgt)                                                                 # line 13
        s = score_fn(src, tgt, outer_seeds(o)) if score_fn is not None else \
            score_heads(fit_heads(views, y, src, outer_seeds(o)), views, tgt)
        if sel is None:
            c = np.asarray(h, dtype=np.int64)
            rep = c.copy()
        else:
            c = candidate_posterior(d, h, s, sel.lam, sel.eta).argmax(1)                         # lines 14-16
            rep = decide(h, c, sel.admitted)
        report[tgt], reference_h[tgt] = rep, h
        out.folds.append(Fold(t, sel, src, tgt, sh, sd, ss, np.asarray(h), np.asarray(d), s, c, rep))
    return out


def make_precomputed_reference(product, folds, target_h, target_d, source_h, source_d) -> Reference:
    """Reference callable that serves saved outputs of any reference classifier.

    target_h (n,), target_d (n, 4): outputs of the reference fitted on S_t, for the recordings of each target t.
    source_h (F, n), source_d (F, n, 4): for outer fold index o (product folds[o]), the outputs of the reference
    fitted on G minus {t, g}, for the recordings of every source product g. The split is recognized from the products
    present in `train` and `test`.
    """
    product = np.asarray(product)
    folds = list(folds)

    def reference(train: np.ndarray, test: np.ndarray):
        test_products = set(product[test].tolist())
        train_products = set(product[train].tolist())
        absent = [g for g in folds if g not in test_products | train_products]
        if not absent and len(test_products) == 1:          # outer call: test is the whole target product t
            return np.asarray(target_h)[test], np.asarray(target_d)[test]
        if len(absent) == 1 and len(test_products) == 1:    # inner call: t absent, test is source product g
            o = folds.index(absent[0])
            return np.asarray(source_h)[o, test], np.asarray(source_d)[o, test]
        raise ValueError("not a whole-product split of the expected form")

    return reference


@dataclass
class Monitor:
    """A deployed VET monitor: frozen configuration, admitted directions, and heads fitted on all known products."""
    selection: Selection | None
    heads: dict

    def report(self, views_new: dict[str, np.ndarray], h_new: np.ndarray, d_new: np.ndarray) -> np.ndarray:
        """Reports for recordings of a new product, given the reference outputs (fitted on all known products)."""
        if self.selection is None:
            return np.asarray(h_new, dtype=np.int64)
        rows = np.arange(len(h_new))
        s = score_heads(self.heads, views_new, rows)
        c = candidate_posterior(d_new, h_new, s, self.selection.lam, self.selection.eta).argmax(1)
        return decide(h_new, c, self.selection.admitted)


def fit_monitor(y, product, views, reference: Reference, *, r_min: int = R_MIN, share: float = SHARE) -> Monitor:
    """Fit VET on all known products (the deployment counterpart of one outer fold, with every product as source)."""
    y, product = np.asarray(y, dtype=np.int64), np.asarray(product)
    rows = np.arange(len(y))
    n_folds = len(set(product.tolist()))
    sh, sd, ss = heldout_source_predictions(y, product, views, reference, n_folds, rows)
    sel = select_configuration(y, product, sd, sh, ss, r_min=r_min, share=share)
    return Monitor(sel, fit_heads(views, y, rows, {k: v for k, v in outer_seeds(n_folds).items()}))
