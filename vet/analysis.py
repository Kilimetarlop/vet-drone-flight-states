"""Held-out source analysis of the correction mechanism (paper Section V-F, Fig. 8(a)-(b), Table VI).

Inside each outer fold, every source product g is held out once more: the configuration, the admitted set, and the
gate threshold are selected on the held-out source predictions of the other source products and applied to g. Each
change that the untested blend proposes on g is then recorded with its confidence, its outcome, and whether the
direction test or the gate admits it. Source labels only; no target label is read.
"""

from __future__ import annotations

import numpy as np

from .candidate import candidate_posterior
from .configuration import select_configuration
from .direction_test import outcomes
from .gate import select_threshold
from .metrics import auc


def heldout_source_changes(y, product, holdout) -> list[dict]:
    y, product = np.asarray(y, dtype=np.int64), np.asarray(product)
    changes = []
    for f in holdout.folds:
        ys, gs = y[f.source_rows], product[f.source_rows]
        for g in sorted(set(gs.tolist())):
            tr, te = gs != g, gs == g
            sel = select_configuration(ys[tr], gs[tr], f.source_d[tr], f.source_h[tr], f.source_scores[tr])
            if sel is None:
                continue
            gamma = select_threshold(ys[tr], gs[tr], f.source_d[tr], f.source_h[tr], f.source_scores[tr],
                                     sel.lam, sel.eta)
            z = candidate_posterior(f.source_d[te], f.source_h[te], f.source_scores[te], sel.lam, sel.eta)
            c, conf, h = z.argmax(1), z.max(1), f.source_h[te]
            out = outcomes(h, c, ys[te])
            for k in np.flatnonzero(c != h):
                changes.append({"target": f.target, "held_out_source": g, "direction": (int(h[k]), int(c[k])),
                                "confidence": float(conf[k]), "outcome": str(out[k]),
                                "test_admits": bool(sel.admitted[h[k], c[k]]),
                                "gate_admits": bool(gamma is not None and conf[k] >= gamma)})
    return changes


def summarize_changes(changes: list[dict]) -> dict:
    """Totals, the repair share and damages of each rule, and how well confidence ranks repairs above damages."""
    def tally(rows):
        return {o: sum(r["outcome"] == o for r in rows) for o in ("repair", "damage", "neutral")}

    rep = np.array([r["confidence"] for r in changes if r["outcome"] == "repair"])
    dam = np.array([r["confidence"] for r in changes if r["outcome"] == "damage"])
    labels = np.concatenate((np.ones(len(rep)), np.zeros(len(dam))))
    out = {"untested": tally(changes),
           "direction_test": tally([r for r in changes if r["test_admits"]]),
           "confidence_gate": tally([r for r in changes if r["gate_admits"]]),
           "confidence_auc_repair_vs_damage": auc(labels, np.concatenate((rep, dam)))}
    for rule in ("untested", "direction_test", "confidence_gate"):
        t = out[rule]
        n = sum(t.values())
        t["repair_share"] = t["repair"] / n if n else float("nan")
    return out


def threshold_sweep(changes: list[dict]) -> list[dict]:
    """Every pooled confidence threshold over the held-out source changes (repairs, damages, neutral admitted)."""
    conf = np.array([r["confidence"] for r in changes])
    out = np.array([r["outcome"] for r in changes])
    rows = []
    for thr in sorted({0.0, *conf.tolist()}):
        keep = conf >= thr
        rows.append({"threshold": thr, **{o: int(np.sum(keep & (out == o))) for o in ("repair", "damage", "neutral")}})
    return rows
