"""Run VET under whole-product holdout on extracted views (paper Algorithm 1).

    python scripts/run_vet.py views.npz [--reference-outputs reference.npz] [--out reports.csv]

The reference classifier is Step 1 of VET. `reference/run_reference.py model` writes the outputs of the paper's
hierarchical reference in this format; any other reference can be passed with --reference-outputs, an .npz with
    event     (n,)       optional: recording file names, used to align rows with the views
    folds     (F,)       product names in outer-fold order (sorted)
    target_h  (n,)       judged state of every recording, from the reference fitted on the other products
    target_d  (n, 4)     four-state posterior of every recording, same fits
    source_h  (F, n)     for fold o: judged state of every source recording, from the reference fitted without
                         folds[o] and without that recording's product
    source_d  (F, n, 4)  the matching posteriors
Without it, the stand-in reference of vet.reference (logistic regression on both views) is fitted inside every split;
its numbers differ from the paper's.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vet import metrics  # noqa: E402
from vet.gate import run_confidence_gate  # noqa: E402
from vet.model import make_precomputed_reference, run_whole_product_holdout  # noqa: E402
from vet.reference import make_logistic_reference  # noqa: E402
from vet.states import ABBREVIATIONS, direction_name  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("views", type=Path)
    ap.add_argument("--reference-outputs", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports.csv"))
    args = ap.parse_args()
    a = np.load(args.views, allow_pickle=False)
    y, product = a["y"].astype(np.int64), a["product"].astype(str)
    views = {"envelope": a["envelope"].astype(np.float64), "stft": a["stft"].astype(np.float64)}
    if args.reference_outputs:
        r = np.load(args.reference_outputs, allow_pickle=False)
        idx = np.arange(len(y))
        if "event" in r.files:                       # align rows by recording file name
            order = {e: i for i, e in enumerate(r["event"].astype(str))}
            idx = np.array([order[e] for e in a["event"].astype(str)])
        reference = make_precomputed_reference(product, r["folds"].astype(str).tolist(), r["target_h"][idx],
                                               r["target_d"][idx], r["source_h"][:, idx], r["source_d"][:, idx])
    else:
        reference = make_logistic_reference(np.hstack((views["envelope"], views["stft"])), y, temperature=3.0)
    res = run_whole_product_holdout(y, product, views, reference)
    gate, _ = run_confidence_gate(y, product, res)
    for name, p in (("reference", res.reference), ("confidence gate", gate), ("VET", res.report)):
        m = metrics.scores(y, p)
        print(f"{name:16s} correct {m['correct']}/{m['total']}  accuracy {100 * m['accuracy']:.2f}  "
              f"balanced accuracy {100 * m['balanced_accuracy']:.2f}  Macro-F1 {100 * m['macro_f1']:.2f}")
    print("VET against the reference:", metrics.paired(y, res.reference, res.report))
    for f in res.folds:
        sel = "reference kept" if f.selection is None else (
            f"lambda={f.selection.lam:.3f} eta={f.selection.eta:.3f} admitted "
            f"{[direction_name(k, l) for k, l in zip(*np.nonzero(f.selection.admitted))]}")
        print(f"  target {f.target}: {sel}")
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "product", "true", "reference", "vet"])
        for i in range(len(y)):
            w.writerow([a["event"][i], product[i], ABBREVIATIONS[y[i]], ABBREVIATIONS[res.reference[i]],
                        ABBREVIATIONS[res.report[i]]])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
