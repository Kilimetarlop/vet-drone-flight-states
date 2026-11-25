"""Compare a fresh run of the reference classifier with the arrays behind the paper (not distributed).

    python reference/verify_reference.py --work work/ --saved-h1 H1.npz --saved-h3 H3.npz --saved-inner inner.npz
                                         [--vet-cache paper_arrays.npz]

Checks the outer folds (horizontal-expert reports, domain-invariant reports and posteriors, final judgments) and the
held-out source products of the strict double exclusion against the saved arrays, then, with --vet-cache, reruns VET on
the fresh reference outputs and compares its 168 reports with the paper's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--saved-h1", type=Path, required=True)
    ap.add_argument("--saved-h3", type=Path, required=True)
    ap.add_argument("--saved-inner", type=Path, required=True)
    ap.add_argument("--vet-cache", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reference_verification.json"))
    args = ap.parse_args()
    outer = np.load(args.work / "reference_outer.npz", allow_pickle=False)
    inner = np.load(args.work / "reference_inner.npz", allow_pickle=False)
    h1 = np.load(args.saved_h1, allow_pickle=False)
    h3 = np.load(args.saved_h3, allow_pickle=False)
    si = np.load(args.saved_inner, allow_pickle=False)
    assert _same(outer["event"].astype(str), h3["event"].astype(str)), "event order differs"
    mask = si["meta_source_mask"].astype(bool)
    rep = {
        "outer": {
            "base_identical": _same(outer["base_prediction"], h1["strict_refit_base_prediction"]),
            "H1_identical": _same(outer["H1_prediction"], h1["H1_prediction"]),
            "meta_H1_identical": _same(outer["meta_H1_prediction"][mask], h1["meta_H1_prediction"][mask]),
            "StrictMain_identical": _same(outer["StrictMain_prediction"], h3["StrictMain_prediction"]),
            "StrictMain_probability_max_abs_diff": float(np.abs(outer["StrictMain_probability"].astype(np.float64)
                                                                - h3["StrictMain_probability"]).max()),
            "H3c_identical": _same(outer["H3c_prediction"], h3["H3c_prediction"]),
            "H3c_differences": int(np.sum(outer["H3c_prediction"] != h3["H3c_prediction"])),
        },
        "inner": {
            name: _same(inner[name][mask], si[name][mask])
            for name in ("meta_StrictMain_prediction", "meta_ConditionPrototype_prediction", "meta_H1_prediction",
                         "meta_H3c_prediction") if name in inner.files and name in si.files
        },
    }
    rep["inner"]["meta_StrictMain_probability_max_abs_diff"] = float(
        np.abs(inner["meta_StrictMain_probability"][mask].astype(np.float64)
               - si["meta_StrictMain_probability"][mask]).max())
    if args.vet_cache:
        from vet.model import make_precomputed_reference, run_whole_product_holdout

        a = np.load(args.vet_cache, allow_pickle=False)
        r = np.load(args.work / "reference_outputs.npz", allow_pickle=False) if (
            args.work / "reference_outputs.npz").exists() else None
        order = {e: i for i, e in enumerate(r["event"].astype(str))}
        idx = np.array([order[e] for e in a["event"].astype(str)])          # align to the paper's event order
        product, folds = a["product"].astype(str), r["folds"].astype(str).tolist()
        ref = make_precomputed_reference(product, folds, r["target_h"][idx], r["target_d"][idx],
                                         r["source_h"][:, idx], r["source_d"][:, idx])
        views = {"envelope": a["envelope"].astype(np.float64), "stft": a["stft"].astype(np.float64)}
        res = run_whole_product_holdout(a["y"].astype(np.int64), product, views, ref)
        rep["vet_on_fresh_reference"] = {"correct": int((res.report == a["y"]).sum()),
                                         "identical_reports": _same(res.report, a["vet_report"])}
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    ok = (rep["outer"]["H3c_identical"] and rep["outer"]["StrictMain_identical"]
          and all(v for k, v in rep["inner"].items() if isinstance(v, bool))
          and rep.get("vet_on_fresh_reference", {}).get("identical_reports", True))
    print("REFERENCE REPRODUCED" if ok else "REFERENCE DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
