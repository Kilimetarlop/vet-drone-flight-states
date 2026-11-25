"""Check this package against the arrays that produced the paper's results.

The arrays (per-recording views, the reference classifier's outputs on target and held-out source products, the head
scores, and every method's reports) come from the authors' pipeline and are not distributed with this repository.
Given them in one .npz file, this script reruns every step of VET with the package and compares:

  heads      head scores refitted from the views against the saved scores (outer folds and held-out source products)
  VET        every target report, and each fold's configuration and admitted directions
  gate       the confidence-gated correction reports
  mechanism  the held-out source changes behind Fig. 8 and the pooled threshold sweep
  thresholds the post-hoc sensitivity rows of Table VI
  statistics Wilson interval, McNemar and sign-flip tests, product bootstrap intervals
  views      (optional, --raw-root) both views re-extracted from raw DRFF-R2 files
  baselines  (optional, --with-baselines) the seven retrained DG methods

Usage: python scripts/verify_paper.py --cache paper_arrays.npz [--raw-root DIR --n-raw 4] [--with-baselines]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vet import analysis, metrics  # noqa: E402
from vet.candidate import CONFIGURATIONS  # noqa: E402
from vet.gate import run_confidence_gate  # noqa: E402
from vet.model import make_precomputed_reference, run_whole_product_holdout  # noqa: E402
from vet.states import direction_name  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path)
    ap.add_argument("--n-raw", type=int, default=4)
    ap.add_argument("--with-baselines", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("verification.json"))
    args = ap.parse_args()
    a = dict(np.load(args.cache, allow_pickle=False))
    y, product, folds = a["y"].astype(np.int64), a["product"].astype(str), a["folds"].astype(str).tolist()
    views = {"envelope": a["envelope"].astype(np.float64), "stft": a["stft"].astype(np.float64)}
    assert folds == sorted(set(product.tolist())), "outer folds must run in sorted product order"
    reference = make_precomputed_reference(product, folds, a["reference_h"], a["reference_d"], a["source_h"],
                                           a["source_d"])
    report: dict = {}
    ok = True

    # ---- heads and VET (Algorithm 1)
    t0 = time.perf_counter()
    res = run_whole_product_holdout(y, product, views, reference)
    outer = np.zeros((len(y), 3))
    worst_inner = 0.0
    for o, f in enumerate(res.folds):
        outer[f.target_rows] = f.scores
        saved = a["source_head_scores"][o, f.source_rows]
        worst_inner = max(worst_inner, float(np.abs(f.source_scores - saved).max()))
    vet_same = bool(np.array_equal(res.report, a["vet_report"]))
    report["heads"] = {"max_abs_diff_target": float(np.abs(outer - a["head_scores"]).max()),
                       "max_abs_diff_heldout_source": worst_inner}
    report["vet"] = {"correct": int((res.report == y).sum()), "identical_reports": vet_same,
                     "folds": [{"target": f.target, "lambda": f.selection.lam, "eta": f.selection.eta,
                                "admitted": [direction_name(k, l) for k, l in zip(*np.nonzero(f.selection.admitted))]}
                               for f in res.folds], "seconds": time.perf_counter() - t0}
    ok &= vet_same and report["heads"]["max_abs_diff_target"] < 1e-6 and worst_inner < 1e-6

    # ---- scores of the reference and VET, statistics
    h = a["reference_h"].astype(np.int64)
    m_ref, m_vet = metrics.scores(y, h), metrics.scores(y, res.report)
    per_product = {g: [int((h == y)[product == g].sum()), int((res.report == y)[product == g].sum())] for g in folds}
    diffs = np.array([v[1] - v[0] for v in per_product.values()])
    report["statistics"] = {
        "reference": {k: m_ref[k] for k in ("correct", "accuracy", "balanced_accuracy", "macro_f1")},
        "vet": {k: m_vet[k] for k in ("correct", "accuracy", "balanced_accuracy", "macro_f1", "recall")},
        "paired_vs_reference": metrics.paired(y, h, res.report),
        "wilson_accuracy": metrics.wilson(m_vet["correct"], len(y)),
        "per_product_correct_reference_vet": per_product,
        "sign_flip_p": metrics.sign_flip_p(diffs),
        "product_bootstrap_95": metrics.product_bootstrap(y, res.report, h, product, products=folds)}

    # ---- confidence gate
    gate, chosen = run_confidence_gate(y, product, res)
    gate_same = bool(np.array_equal(gate, a["gate_report"]))
    report["gate"] = {"correct": int((gate == y).sum()), "identical_reports": gate_same, "thresholds": chosen,
                      "vet_vs_gate": metrics.paired(y, gate, res.report)}
    ok &= gate_same

    # ---- held-out source mechanism (Fig. 8(a)-(b)) and pooled thresholds
    changes = analysis.heldout_source_changes(y, product, res)
    summary = analysis.summarize_changes(changes)
    sweep = analysis.threshold_sweep(changes)
    t = summary["direction_test"]
    at_repairs = [s for s in sweep if s["repair"] >= t["repair"]]
    at_damages = [s for s in sweep if s["damage"] <= t["damage"]]
    report["mechanism"] = {"n_changes": len(changes), **summary,
                           "fewest_damages_at_test_repairs": min(s["damage"] for s in at_repairs),
                           "most_repairs_at_test_damages": max(s["repair"] for s in at_damages),
                           "largest_net_gain": max(s["repair"] - s["damage"] for s in sweep)}

    # ---- post-hoc sensitivity (Table VI)
    grid = []
    for r_min in (1, 2, 3):
        for share in (1 / 2, 3 / 5, 2 / 3, 3 / 4):
            p = run_whole_product_holdout(y, product, views, reference, r_min=r_min, share=share).report
            m = metrics.scores(y, p)
            grid.append({"r_min": r_min, "share": share, "correct": m["correct"], "balanced_accuracy": m["balanced_accuracy"]})
    fixed = []
    for lam, eta in CONFIGURATIONS:
        p = run_whole_product_holdout(y, product, views, reference, fixed=(lam, eta)).report
        m = metrics.scores(y, p)
        fixed.append({"lambda": lam, "eta": eta, "correct": m["correct"], "balanced_accuracy": m["balanced_accuracy"]})
    report["sensitivity"] = {"thresholds": grid, "fixed_configurations": fixed}

    # ---- optional: views from raw recordings
    if args.raw_root:
        from vet.data import inventory
        from vet.views import recording_views

        rows = {r.event: r for r in inventory(args.raw_root)}
        pick = np.linspace(0, len(y) - 1, args.n_raw, dtype=int)
        diff = {"envelope": 0.0, "stft": 0.0, "events": []}
        for i in pick:
            v = recording_views(rows[str(a["event"][i])].path)
            diff["envelope"] = max(diff["envelope"], float(np.abs(v["envelope"] - a["envelope"][i]).max()))
            diff["stft"] = max(diff["stft"], float(np.abs(v["stft"] - a["stft"][i]).max()))
            diff["events"].append(str(a["event"][i]))
        report["views"] = diff

    # ---- optional: retrained DG baselines (Table III)
    if args.with_baselines:
        from vet.baselines import METHODS, run_baseline

        x = np.hstack((a["envelope"], a["stft"])).astype(np.float64)
        report["baselines"] = {}
        for method in METHODS:
            p, chosen_dg = run_baseline(method, x, y, product)
            same = bool(np.array_equal(p, a[f"dg_{method}"]))
            report["baselines"][method] = {"correct": int((p == y).sum()), "identical_reports": same}
            ok &= same

    report["all_checks_passed"] = bool(ok)
    args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("heads", "vet", "gate")}, indent=1, default=float)[:3000])
    print("mechanism:", {k: report["mechanism"][k] for k in ("n_changes", "untested", "direction_test",
                                                               "confidence_gate", "confidence_auc_repair_vs_damage")})
    print("sensitivity correct:", [g["correct"] for g in grid], [f["correct"] for f in fixed])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
