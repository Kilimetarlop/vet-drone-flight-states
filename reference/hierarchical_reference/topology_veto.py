"""Final reference judgment: the domain-invariant veto of cruise labels and the Hover override of the
condition-prototype branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np

from .horizontal_expert import _five_metrics, _validate_inventory
from .router import MultiViewBaseCache


STRICT_BRANCH = "DomainInvariant"


AGREEMENT_BRANCH = "ConditionPrototype"


NON_HORIZONTAL = (2, 3, 4)


VGAP_THRESHOLDS = (
    float("-inf"),
    -0.15,
    -0.10,
    -0.075,
    -0.05,
    -0.025,
    0.0,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    float("inf"),
)


def load_h1_artifact(
    path: Path,
    *,
    event: np.ndarray,
    group: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    required = {
        "event",
        "identity",
        "target",
        "H1_prediction",
        "meta_outer_held_groups",
        "meta_source_mask",
        "meta_H1_prediction",
        "method_version",
    }
    with np.load(path, allow_pickle=False) as cache:
        if required - set(cache.files):
            raise RuntimeError("H1 artifact lacks strict meta predictions")
        loaded = {name: np.asarray(cache[name]).copy() for name in cache.files}
    if (
        not np.array_equal(loaded["event"].astype(str), event)
        or not np.array_equal(loaded["identity"].astype(str), group)
        or not np.array_equal(loaded["target"].astype(np.int64), target)
    ):
        raise RuntimeError("H1 artifact row alignment differs")
    groups = tuple(sorted(np.unique(group).astype(str).tolist()))
    if tuple(loaded["meta_outer_held_groups"].astype(str).tolist()) != groups:
        raise RuntimeError("H1 meta outer-group order differs")
    source_mask = loaded["meta_source_mask"].astype(bool)
    meta_h1 = loaded["meta_H1_prediction"].astype(np.int64)
    if source_mask.shape != (8, len(event)) or meta_h1.shape != source_mask.shape:
        raise RuntimeError("H1 strict meta matrix shape differs")
    for fold_index, held in enumerate(groups):
        expected = group != held
        if (
            not np.array_equal(source_mask[fold_index], expected)
            or np.any(meta_h1[fold_index, expected] < 0)
            or np.any(meta_h1[fold_index, ~expected] != -1)
        ):
            raise RuntimeError("H1 source-meta exclusion audit failed")
    loaded["H1_prediction"] = loaded["H1_prediction"].astype(np.int64)
    loaded["meta_source_mask"] = source_mask
    loaded["meta_H1_prediction"] = meta_h1
    return loaded


def strict_topology_gap(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 5:
        raise ValueError("StrictMain probability must be records x five classes")
    if not np.isfinite(value).all():
        raise ValueError("StrictMain probability contains non-finite values")
    return value[:, 2:].max(axis=1) - value[:, :2].max(axis=1)


def threshold_label(threshold: float) -> str | float:
    if np.isneginf(threshold):
        return "all_veto"
    if np.isposinf(threshold):
        return "no_veto"
    return float(threshold)


def apply_strict_veto(
    h1_prediction: np.ndarray,
    strict_prediction: np.ndarray,
    strict_probability: np.ndarray,
    *,
    threshold: float = float("-inf"),
    agreement_prediction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h1 = np.asarray(h1_prediction, dtype=np.int64)
    strict = np.asarray(strict_prediction, dtype=np.int64)
    if h1.shape != strict.shape:
        raise ValueError("H1 and StrictMain predictions do not align")
    gap = strict_topology_gap(strict_probability)
    allowed = (
        np.isin(h1, (0, 1))
        & np.isin(strict, NON_HORIZONTAL)
        & (gap >= float(threshold))
    )
    if agreement_prediction is not None:
        agreement = np.asarray(agreement_prediction, dtype=np.int64)
        if agreement.shape != h1.shape:
            raise ValueError("agreement prediction does not align")
        allowed &= agreement == strict
    output = h1.copy()
    output[allowed] = strict[allowed]
    return output, allowed & (output != h1)


def apply_hover_specialist(
    prediction: np.ndarray,
    condition_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ConditionPrototype is allowed to emit HOVER only, never another class."""

    base = np.asarray(prediction, dtype=np.int64)
    condition = np.asarray(condition_prediction, dtype=np.int64)
    if base.shape != condition.shape:
        raise ValueError("hover specialist inputs do not align")
    allowed = condition == 2
    output = base.copy()
    output[allowed] = 2
    return output, allowed & (output != base)


def select_binary_switch(
    source_target: np.ndarray,
    off_prediction: np.ndarray,
    on_prediction: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    scores = {
        "off": _five_metrics(source_target, off_prediction),
        "on": _five_metrics(source_target, on_prediction),
    }
    gain_f1 = scores["on"]["macro_f1"] - scores["off"]["macro_f1"]
    gain_acc = scores["on"]["accuracy"] - scores["off"]["accuracy"]
    enabled = bool(
        gain_f1 > 1.0e-12
        or (abs(gain_f1) <= 1.0e-12 and gain_acc > 1.0e-12)
    )
    return enabled, {"selected": "on" if enabled else "off", "scores": scores}


def select_vgap_threshold(
    source_target: np.ndarray,
    candidate_prediction: Mapping[float, np.ndarray],
) -> tuple[float, list[dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for threshold, prediction in candidate_prediction.items():
        metrics = _five_metrics(source_target, prediction)
        scored.append((float(threshold), metrics))
        rows.append(
            {
                "threshold": threshold_label(threshold),
                **metrics,
            }
        )
    ranked = sorted(
        scored,
        key=lambda item: (
            -item[1]["macro_f1"],
            -item[1]["accuracy"],
            -item[0],
        ),
    )
    return float(ranked[0][0]), rows


def _collect_source_topology(
    *,
    held_group: str,
    groups: tuple[str, ...],
    group: np.ndarray,
    base_cache: MultiViewBaseCache,
) -> dict[str, Any]:
    source_groups = tuple(name for name in groups if name != held_group)
    source_indices = np.flatnonzero(group != held_group)
    strict_prediction = np.full(len(group), -1, dtype=np.int64)
    strict_probability = np.full((len(group), 5), np.nan, dtype=np.float32)
    agreement_prediction = np.full(len(group), -1, dtype=np.int64)
    fit_sets = {}
    for source_group in source_groups:
        fit_groups = tuple(
            name for name in groups if name not in (held_group, source_group)
        )
        evidence = base_cache.get(fit_groups, source_group)
        local = np.flatnonzero(group == source_group)
        strict_prediction[local] = evidence[STRICT_BRANCH].prediction
        strict_probability[local] = evidence[STRICT_BRANCH].probability
        agreement_prediction[local] = evidence[AGREEMENT_BRANCH].prediction
        fit_sets[source_group] = list(fit_groups)
    if (
        np.any(strict_prediction[source_indices] < 0)
        or not np.isfinite(strict_probability[source_indices]).all()
        or np.any(agreement_prediction[source_indices] < 0)
    ):
        raise RuntimeError("strict source topology evidence assembly failed")
    return {
        "source_groups": source_groups,
        "source_indices": source_indices,
        "strict_prediction": strict_prediction[source_indices],
        "strict_probability": strict_probability[source_indices],
        "agreement_prediction": agreement_prediction[source_indices],
        "fit_sets": fit_sets,
    }


def build_outer_fold_unscored(
    *,
    fold_index: int,
    held_group: str,
    groups: tuple[str, ...],
    group: np.ndarray,
    target: np.ndarray,
    h1: Mapping[str, np.ndarray],
    base_cache: MultiViewBaseCache,
) -> dict[str, Any]:
    """Create one H3 fold; no held target label is indexed for any choice."""

    source = _collect_source_topology(
        held_group=held_group,
        groups=groups,
        group=group,
        base_cache=base_cache,
    )
    source_indices = source["source_indices"]
    source_h1 = h1["meta_H1_prediction"][fold_index, source_indices]
    source_h3b_candidates = {
        threshold: apply_strict_veto(
            source_h1,
            source["strict_prediction"],
            source["strict_probability"],
            threshold=threshold,
        )[0]
        for threshold in VGAP_THRESHOLDS
    }
    selected_threshold, threshold_audit = select_vgap_threshold(
        target[source_indices], source_h3b_candidates,
    )

    target_indices = np.flatnonzero(group == held_group)
    target_evidence = base_cache.get(source["source_groups"], held_group)
    target_strict = target_evidence[STRICT_BRANCH].prediction
    target_probability = target_evidence[STRICT_BRANCH].probability
    target_agreement = target_evidence[AGREEMENT_BRANCH].prediction
    target_h1 = h1["H1_prediction"][target_indices]
    h3a, h3a_changed = apply_strict_veto(
        target_h1, target_strict, target_probability,
    )
    h3b, h3b_changed = apply_strict_veto(
        target_h1,
        target_strict,
        target_probability,
        threshold=selected_threshold,
    )
    source_h3a, _ = apply_strict_veto(
        source_h1, source["strict_prediction"], source["strict_probability"],
    )
    source_h3c_on, _ = apply_hover_specialist(
        source_h3a, source["agreement_prediction"],
    )
    h3c_enabled, h3c_selection = select_binary_switch(
        target[source_indices], source_h3a, source_h3c_on,
    )
    h3c_fixed_on, _ = apply_hover_specialist(h3a, target_agreement)
    h3c = h3c_fixed_on if h3c_enabled else h3a.copy()
    h3c_changed = h3c != target_h1
    source_h3c = source_h3c_on if h3c_enabled else source_h3a.copy()
    return {
        "held_group": held_group,
        "source_groups": list(source["source_groups"]),
        "source_fit_sets": source["fit_sets"],
        "source_indices": source_indices,
        "source_strict_prediction": source["strict_prediction"],
        "source_strict_probability": source["strict_probability"],
        "source_agreement_prediction": source["agreement_prediction"],
        "source_H3a_prediction": source_h3a,
        "source_H3b_prediction": source_h3b_candidates[selected_threshold],
        "source_H3c_prediction": source_h3c,
        "target_indices": target_indices,
        "target_strict_prediction": target_strict,
        "target_strict_probability": target_probability,
        "target_agreement_prediction": target_agreement,
        "target_H3a_prediction": h3a,
        "target_H3b_prediction": h3b,
        "target_H3c_prediction": h3c,
        "target_H3c_fixed_on_posthoc_ceiling_prediction": h3c_fixed_on,
        "target_H3a_changed": h3a_changed,
        "target_H3b_changed": h3b_changed,
        "target_H3c_changed": h3c_changed,
        "H3b_selected_threshold": selected_threshold,
        "H3b_source_selection": threshold_audit,
        "H3c_hover_specialist_selection": h3c_selection,
        "outer_target_labels_read_for_fit_or_selection": False,
    }


def generate_outer_predictions_unscored(
    *,
    event: np.ndarray,
    group: np.ndarray,
    target: np.ndarray,
    h1: Mapping[str, np.ndarray],
    base_cache: MultiViewBaseCache,
) -> dict[str, Any]:
    groups = _validate_inventory(event, group, target)
    target_output = {
        name: np.full(len(target), -1, dtype=np.int64)
        for name in (
            "strict",
            "agreement",
            "H3a",
            "H3b",
            "H3c",
            "H3c_fixed_on_posthoc_ceiling",
        )
    }
    target_probability = np.full((len(target), 5), np.nan, dtype=np.float32)
    target_gap = np.full(len(target), np.nan, dtype=np.float32)
    target_threshold = np.full(len(target), np.nan, dtype=np.float64)
    target_h3c_enabled = np.zeros(len(target), dtype=bool)
    target_changed = {
        name: np.zeros(len(target), dtype=bool) for name in ("H3a", "H3b", "H3c")
    }
    meta_output = {
        name: np.full((len(groups), len(target)), -1, dtype=np.int64)
        for name in ("strict", "agreement", "H3a", "H3b", "H3c")
    }
    meta_probability = np.full(
        (len(groups), len(target), 5), np.nan, dtype=np.float32,
    )
    meta_source_mask = np.zeros((len(groups), len(target)), dtype=bool)
    folds: list[dict[str, Any]] = []
    for fold_index, held_group in enumerate(groups):
        fold = build_outer_fold_unscored(
            fold_index=fold_index,
            held_group=held_group,
            groups=groups,
            group=group,
            target=target,
            h1=h1,
            base_cache=base_cache,
        )
        test = fold["target_indices"]
        source = fold["source_indices"]
        if np.any(target_output["H3a"][test] >= 0):
            raise RuntimeError("H3 outer target rows overlap")
        target_output["strict"][test] = fold["target_strict_prediction"]
        target_output["agreement"][test] = fold["target_agreement_prediction"]
        target_probability[test] = fold["target_strict_probability"]
        target_gap[test] = strict_topology_gap(fold["target_strict_probability"])
        for name in ("H3a", "H3b", "H3c"):
            target_output[name][test] = fold[f"target_{name}_prediction"]
            target_changed[name][test] = fold[f"target_{name}_changed"]
            meta_output[name][fold_index, source] = fold[
                f"source_{name}_prediction"
            ]
        target_output["H3c_fixed_on_posthoc_ceiling"][test] = fold[
            "target_H3c_fixed_on_posthoc_ceiling_prediction"
        ]
        target_threshold[test] = fold["H3b_selected_threshold"]
        target_h3c_enabled[test] = (
            fold["H3c_hover_specialist_selection"]["selected"] == "on"
        )
        meta_source_mask[fold_index, source] = True
        meta_output["strict"][fold_index, source] = fold[
            "source_strict_prediction"
        ]
        meta_output["agreement"][fold_index, source] = fold[
            "source_agreement_prediction"
        ]
        meta_probability[fold_index, source] = fold["source_strict_probability"]
        folds.append(
            {
                key: (
                    threshold_label(value)
                    if key == "H3b_selected_threshold"
                    else value
                )
                for key, value in fold.items()
                if key
                not in {
                    "source_indices",
                    "source_strict_prediction",
                    "source_strict_probability",
                    "source_agreement_prediction",
                    "source_H3a_prediction",
                    "source_H3b_prediction",
                    "source_H3c_prediction",
                    "target_indices",
                    "target_strict_prediction",
                    "target_strict_probability",
                    "target_agreement_prediction",
                    "target_H3a_prediction",
                    "target_H3b_prediction",
                    "target_H3c_prediction",
                    "target_H3c_fixed_on_posthoc_ceiling_prediction",
                    "target_H3a_changed",
                    "target_H3b_changed",
                    "target_H3c_changed",
                }
            }
        )
        print(
            f"H3 fold={fold_index + 1}/8 held={held_group} "
            f"H3b_threshold={threshold_label(fold['H3b_selected_threshold'])} "
            "outer_target_unscored",
            flush=True,
        )
    if (
        any(np.any(value < 0) for value in target_output.values())
        or not np.isfinite(target_probability).all()
        or not np.isfinite(target_gap).all()
        or any(np.sum(meta_source_mask[index]) != 147 for index in range(8))
    ):
        raise RuntimeError("H3 predictions did not freeze completely")
    return {
        **target_output,
        "strict_probability": target_probability,
        "strict_vertical_horizontal_gap": target_gap,
        "H3b_threshold": target_threshold,
        "H3c_hover_specialist_enabled": target_h3c_enabled,
        "H3a_changed": target_changed["H3a"],
        "H3b_changed": target_changed["H3b"],
        "H3c_changed": target_changed["H3c"],
        "meta_outer_held_groups": np.asarray(groups),
        "meta_source_mask": meta_source_mask,
        "meta_strict_prediction": meta_output["strict"],
        "meta_strict_probability": meta_probability,
        "meta_agreement_prediction": meta_output["agreement"],
        "meta_H3a_prediction": meta_output["H3a"],
        "meta_H3b_prediction": meta_output["H3b"],
        "meta_H3c_prediction": meta_output["H3c"],
        "folds": folds,
        "all_outer_predictions_frozen": True,
    }
