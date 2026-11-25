"""Run the paper's hierarchical reference classifier (Step 1 of VET, Section IV-B) and export its outputs for VET.

    python reference/run_reference.py features --dataset-root DRFF-R2/dataset1-single_drone_states --work work/
    python reference/run_reference.py model --work work/ --out reference_outputs.npz

`features` runs the five raw-data builders (resumable; they read about 100 GB in all) and the four derived steps:
    event tokens -> slow-time energy -> slow-time motion features and the periodic-observability gate
    micro-IQ windows -> scan and STFT recording features (the 4948-d fused view of the domain-invariant branch)
    dense spectrogram -> condition descriptors          spectral activity -> spectral summaries and aligned tracks
`model` fits, for every whole-product fold, the four branches (domain-invariant SVM, condition prototypes, nested
spectral SVM, MultiRocket-Hydra ridge), their plurality vote, the horizontal expert (straight against curved cruise),
the domain-invariant veto and the Hover override; then, with strict double exclusion, the same procedure for every
held-out source product. It writes the format that `scripts/run_vet.py --reference-outputs` reads:
    event, folds, target_h, target_d (four states), source_h, source_d, plus the five-class arrays.

The fitting code in `hierarchical_reference/` is the authors' research code, copied unchanged except for imports and
the dataset-path default; development notes in its comments were removed. Expect about one to two hours for `model` on
a desktop CPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED = 20_260_829
FIVE_TO_FOUR = np.array([0, 0, 1, 2, 3], dtype=np.int64)
CACHES = {
    "domain_features": "drff_r2_dataset1_behavior5_features_v1.npz",
    "metadata": "drff_r2_dataset1_behavior5_micro_iq_f16.npz",
    "dense_cache": "drff_r2_dataset1_behavior5_dense_v1.npz",
    "spectral_trajectory": "drff_r2_dataset1_behavior5_spectral_activity_v1.npz",
    "spectral_features": "drff_r2_dataset1_behavior5_spectral_activity_features_v1.npz",
    "slow_features": "slow_time_motion/record_features.npz",
    "c2_report": "periodic_observability/report.json",
}


def _cli(module, argv):
    """Run one vendored builder through its own command line (its main() parses the arguments and runs it)."""
    saved = sys.argv
    sys.argv = [module.__name__] + [str(a) for a in argv]
    try:
        module.main()
    finally:
        sys.argv = saved


def features(args) -> None:
    from hierarchical_reference import (build_dense_spectrogram, build_event_tokens, build_micro_iq,
                                        build_slow_time_energy, build_spectral_activity, periodic_observability,
                                        scan_stft_features, slow_time_features, spectral_activity_features)
    from hierarchical_reference.slow_time_motion import VIBRATION_FEATURE_NAMES

    w, root, dev = args.work, args.dataset_root, args.device
    w.mkdir(parents=True, exist_ok=True)
    tokens = w / "drff_r2_dataset1_behavior5_event_tokens_v2.npz"
    energy = w / "drff_r2_dataset1_behavior5_slow_time_energy_v3.npz"
    _cli(build_event_tokens, [root, "--partial-dir", w / "event_tokens.partial", "--output", tokens,
                              "--report", w / "event_tokens_report.json", "--device", dev])
    _cli(build_slow_time_energy, [root, "--event-cache", tokens, "--partial-dir", w / "slow_time_energy.partial",
                                  "--output", energy, "--report", w / "slow_time_energy_report.json"])
    _cli(build_micro_iq, [root, "--partial-dir", w / "micro_iq.partial", "--output-cache", w / CACHES["metadata"]])
    _cli(build_dense_spectrogram, [root, "--partial-dir", w / "dense.partial", "--output", w / CACHES["dense_cache"],
                                   "--device", dev])
    _cli(build_spectral_activity, [root, "--partial-dir", w / "spectral_activity.partial",
                                   "--output", w / CACHES["spectral_trajectory"], "--device", dev])
    scan_stft_features.load_or_build_features(w / CACHES["metadata"], w / CACHES["domain_features"],
                                              scan_stft_features._device(dev), False)
    spectral_activity_features.load_or_build_feature_cache(w / CACHES["spectral_trajectory"],
                                                           w / CACHES["spectral_features"])
    source = slow_time_features.load_slow_time_cache(energy)
    record = slow_time_features.build_record_features(source)
    (w / "slow_time_motion").mkdir(exist_ok=True)
    np.savez_compressed(w / CACHES["slow_features"], event=source["event"], identity=source["identity"],
                        class_index=source["target"], B1_feature_names=np.asarray(VIBRATION_FEATURE_NAMES),
                        B1_irregular_vibration=record["B1_irregular_vibration"],
                        method_version=np.asarray(slow_time_features.METHOD_VERSION))
    _cli(periodic_observability, ["--input", energy, "--output", w / "periodic_observability"])
    print("features written to", w)


def _paths(args) -> argparse.Namespace:
    return argparse.Namespace(**{k: getattr(args, k) or args.work / v for k, v in CACHES.items()})


def _load(paths):
    from hierarchical_reference import horizontal_expert, router

    data = router._load_all(paths)
    event = np.asarray(data["event"]).astype(str)
    group = np.asarray(data["group"]).astype(str)
    target = np.asarray(data["target"], dtype=np.int64)
    horizontal_expert._validate_inventory(event, group, target)
    periodic = horizontal_expert.load_periodic_inputs(paths.slow_features, paths.c2_report, event=event,
                                                      group=group, target=target)
    return data, periodic, event, group, target


def model_outer(args) -> None:
    """Outer folds: reference judgments and posteriors of every target product (fitted on its source products)."""
    from hierarchical_reference import horizontal_expert, router, topology_veto

    paths = _paths(args)
    data, periodic, event, group, target = _load(paths)
    base = router.MultiViewBaseCache(domain_views=data["domain_views"], dense_feature=data["dense_feature"],
                                     condition=data["condition"], spectral_views=data["spectral_views"],
                                     aligned_trajectory=data["aligned_trajectory"], event=event, group=group,
                                     target=target, seed=args.seed)
    expert = horizontal_expert.HorizontalExpertCache(representations=periodic["representations"], group=group,
                                                     target=target)
    h1 = horizontal_expert.generate_outer_predictions_unscored(
        event=event, group=group, target=target, c2_gate=periodic["c2_gate"],
        representations=periodic["representations"], base_cache=base, expert_cache=expert)
    h3 = topology_veto.generate_outer_predictions_unscored(
        event=event, group=group, target=target,
        h1={"H1_prediction": h1["H1"], "meta_H1_prediction": h1["meta_H1_prediction"]}, base_cache=base)
    np.savez_compressed(args.work / "reference_outer.npz", event=event, identity=group, target=target,
                        folds=h3["meta_outer_held_groups"], base_prediction=h1["base"], H1_prediction=h1["H1"],
                        meta_H1_prediction=h1["meta_H1_prediction"], H3c_prediction=h3["H3c"],
                        StrictMain_prediction=h3["strict"], StrictMain_probability=h3["strict_probability"])
    print("outer folds written:", args.work / "reference_outer.npz")


def model_inner(args) -> None:
    """Held-out source products with strict double exclusion (selection on the six products left)."""
    from hierarchical_reference import double_exclusion

    paths = _paths(args)
    data, periodic, event, group, target = _load(paths)
    out = double_exclusion._generate(data=data, periodic=periodic, seed=args.seed, poison_audit=False)
    np.savez_compressed(args.work / "reference_inner.npz", event=event, identity=group, target=target,
                        **{k: np.asarray(v) for k, v in out["arrays"].items()})
    print("held-out source products written:", args.work / "reference_inner.npz")


def model(args) -> None:
    if not (args.work / "reference_outer.npz").exists():
        model_outer(args)
    if not (args.work / "reference_inner.npz").exists():
        # the double-exclusion step ran with single-threaded numerical libraries; keep that for exact reproduction
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1"}
        cmd = [sys.executable, str(Path(__file__).resolve()), "model-inner", "--work", str(args.work),
               "--seed", str(args.seed)]
        for k in CACHES:
            if getattr(args, k):
                cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        subprocess.run(cmd, check=True, env=env)
    outer = np.load(args.work / "reference_outer.npz", allow_pickle=False)
    inner = np.load(args.work / "reference_inner.npz", allow_pickle=False)
    folds = outer["folds"].astype(str)
    mask = inner["meta_source_mask"].astype(bool)
    source_h = np.zeros(mask.shape, dtype=np.int64)
    source_d = np.zeros(mask.shape + (4,))
    for o in range(len(folds)):
        m = mask[o]
        source_h[o, m] = FIVE_TO_FOUR[inner["meta_H3c_prediction"][o, m]]
        source_d[o, m] = _four_state(inner["meta_StrictMain_probability"][o, m])
    np.savez_compressed(args.out, event=outer["event"], product=outer["identity"], folds=folds,
                        target_h=FIVE_TO_FOUR[outer["H3c_prediction"]],
                        target_d=_four_state(outer["StrictMain_probability"]), source_h=source_h, source_d=source_d,
                        source_mask=mask, target_h5=outer["H3c_prediction"],
                        target_q5=outer["StrictMain_probability"], source_h5=inner["meta_H3c_prediction"],
                        source_q5=inner["meta_StrictMain_probability"])
    print("reference outputs for scripts/run_vet.py:", args.out)


def _four_state(q5: np.ndarray) -> np.ndarray:
    """Eq. (4): both cruise classes form Horizontal; floored at 1e-8 and renormalized as in VET."""
    q5 = np.asarray(q5, dtype=np.float64)
    p = np.clip(np.column_stack((q5[:, 0] + q5[:, 1], q5[:, 2:])), 1.0e-8, None)
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    f = sub.add_parser("features")
    f.add_argument("--dataset-root", type=Path, required=True)
    f.add_argument("--work", type=Path, required=True)
    f.add_argument("--device", default="auto")
    for name in ("model", "model-outer", "model-inner"):
        m = sub.add_parser(name)
        m.add_argument("--work", type=Path, required=True)
        m.add_argument("--out", type=Path, default=Path("reference_outputs.npz"))
        m.add_argument("--seed", type=int, default=SEED)
        for k in CACHES:   # override any input (by default every cache is read from --work)
            m.add_argument(f"--{k.replace('_', '-')}", type=Path)
    args = ap.parse_args()
    {"features": features, "model": model, "model-outer": model_outer, "model-inner": model_inner}[args.step](args)


if __name__ == "__main__":
    main()
