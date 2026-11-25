"""Horizontal expert: a ridge classifier on slow-time features that separates straight from curved
cruise, with every choice made on held-out source products.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .dataset import BEHAVIOR5_CLASSES
from .router import MultiViewBaseCache, plurality_condition_tie


EXPECTED_RECORDINGS = 168


EXPECTED_GROUPS = 8


EXPECTED_PER_GROUP = 21


EXPECTED_COUNTS = (48, 48, 24, 24, 24)


HORIZONTAL_LABELS = (0, 1)


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


H2_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, float("inf"))


C2_ADVANTAGE_GATE_DB = 3.0


C2_DRIFT_GATE_PERCENT = 10.0


@dataclass(frozen=True)
class ExpertEvidence:
    prediction: np.ndarray
    decision: np.ndarray
    reliability: np.ndarray
    training_margin_scale: float

    def validate(self, rows: int) -> None:
        if self.prediction.shape != (rows,):
            raise RuntimeError("horizontal prediction shape drift")
        if self.decision.shape != (rows,) or self.reliability.shape != (rows,):
            raise RuntimeError("horizontal decision shape drift")
        if np.any(~np.isin(self.prediction, HORIZONTAL_LABELS)):
            raise RuntimeError("horizontal expert predicted outside LINEAR/CURVED")
        if not np.isfinite(self.decision).all() or not np.isfinite(self.reliability).all():
            raise RuntimeError("horizontal expert produced non-finite evidence")
        if self.training_margin_scale <= 0.0:
            raise RuntimeError("invalid training margin scale")


def _five_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(BEHAVIOR5_CLASSES), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=labels, zero_division=0,
    )
    return {
        "correct": int(np.sum(target == prediction)),
        "recordings": int(len(target)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(
            f1_score(target, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            target, prediction, labels=labels,
        ).astype(int).tolist(),
        "per_class": {
            BEHAVIOR5_CLASSES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def _horizontal_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    value = np.asarray(target, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    mask = np.isin(value, HORIZONTAL_LABELS)
    if not np.any(mask):
        raise ValueError("horizontal metrics require LINEAR/CURVED rows")
    horizontal_target = value[mask]
    horizontal_prediction = predicted[mask]
    labels = np.asarray(HORIZONTAL_LABELS, dtype=np.int64)
    return {
        "correct": int(np.sum(horizontal_target == horizontal_prediction)),
        "recordings": int(np.sum(mask)),
        "accuracy": float(accuracy_score(horizontal_target, horizontal_prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(horizontal_target, horizontal_prediction)
        ),
        "macro_f1": float(
            f1_score(
                horizontal_target,
                horizontal_prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            horizontal_target, horizontal_prediction, labels=labels,
        ).astype(int).tolist(),
    }


def _validate_inventory(
    event: np.ndarray, group: np.ndarray, target: np.ndarray,
) -> tuple[str, ...]:
    groups = tuple(sorted(np.unique(group).astype(str).tolist()))
    counts = tuple(int(np.sum(target == index)) for index in range(5))
    group_counts = tuple(int(np.sum(group == name)) for name in groups)
    if (
        len(event) != EXPECTED_RECORDINGS
        or len(groups) != EXPECTED_GROUPS
        or group_counts != (EXPECTED_PER_GROUP,) * EXPECTED_GROUPS
        or counts != EXPECTED_COUNTS
    ):
        raise RuntimeError("DRFF-R2 behavior-5 inventory drift")
    return groups


def load_periodic_inputs(
    slow_features_path: Path,
    c2_report_path: Path,
    *,
    event: np.ndarray,
    group: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Load only fixed, recording-local periodic coordinates and C2 gates."""

    with np.load(slow_features_path, allow_pickle=False) as cache:
        required = {
            "event",
            "identity",
            "class_index",
            "B1_feature_names",
            "B1_irregular_vibration",
            "method_version",
        }
        if required - set(cache.files):
            raise RuntimeError("slow-time record feature artifact is incomplete")
        slow_event = np.asarray(cache["event"]).astype(str)
        slow_group = np.asarray(cache["identity"]).astype(str)
        slow_target = np.asarray(cache["class_index"], dtype=np.int64)
        b1_names = np.asarray(cache["B1_feature_names"]).astype(str)
        b1 = np.asarray(cache["B1_irregular_vibration"], dtype=np.float32)
        slow_method = str(np.asarray(cache["method_version"]).reshape(-1)[0])
    if (
        not np.array_equal(slow_event, event)
        or not np.array_equal(slow_group, group)
        or not np.array_equal(slow_target, target)
        or b1.shape != (len(event), len(b1_names))
        or not np.isfinite(b1).all()
    ):
        raise RuntimeError("slow-time feature alignment or finiteness failure")

    payload = json.loads(c2_report_path.read_text(encoding="utf-8"))
    statistics = payload.get("record_statistics", [])
    if len(statistics) != len(event):
        raise RuntimeError("C2 record inventory differs")
    by_event = {str(row["event"]): row for row in statistics}
    if set(by_event) != set(event.tolist()):
        raise RuntimeError("C2 event inventory differs")
    rows = [by_event[str(name)] for name in event]
    for index, row in enumerate(rows):
        if (
            str(row["identity"]) != str(group[index])
            or int(row["behavior_index"]) != int(target[index])
        ):
            raise RuntimeError("C2 row alignment failed")

    def column(name: str) -> np.ndarray:
        value = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite C2 coordinate: {name}")
        return value

    c2_names = np.asarray(
        (
            "real_peak_hz",
            "log10_real_peak_power",
            "log10_shuffle_peak_power_median",
            "log10_shuffle_peak_power_q95",
            "real_vs_shuffle_median_peak_advantage_db",
            "negative_log10_shuffle_empirical_p",
            "delete_10pct_peak_drift_median_percent",
            "delete_10pct_peak_drift_q90_percent",
            "delete_10pct_peak_same_bin_fraction",
            "q90_gap_nyquist_proxy_hz",
            "spectrum_high_hz",
            "log1p_burst_samples",
            "full_10_90hz_eligible",
        )
    )
    c2 = np.column_stack(
        (
            column("real_peak_hz"),
            np.log10(np.maximum(column("real_peak_power"), 1.0e-12)),
            np.log10(np.maximum(column("shuffle_peak_power_median"), 1.0e-12)),
            np.log10(np.maximum(column("shuffle_peak_power_q95"), 1.0e-12)),
            column("real_vs_shuffle_median_peak_advantage_db"),
            -np.log10(np.maximum(column("shuffle_empirical_p"), 1.0e-12)),
            column("delete_10pct_peak_drift_median_percent"),
            column("delete_10pct_peak_drift_q90_percent"),
            column("delete_10pct_peak_same_bin_fraction"),
            column("q90_gap_nyquist_proxy_hz"),
            column("spectrum_high_hz"),
            np.log1p(column("burst_samples")),
            column("full_10_90hz_eligible"),
        )
    ).astype(np.float32)

    eligible = np.asarray(
        [bool(row["full_10_90hz_eligible"]) for row in rows], dtype=bool,
    )
    advantage = column("real_vs_shuffle_median_peak_advantage_db")
    drift = column("delete_10pct_peak_drift_median_percent")
    c2_gate = (
        eligible
        & (advantage >= C2_ADVANTAGE_GATE_DB)
        & (drift <= C2_DRIFT_GATE_PERCENT)
    )
    reported_gate = eligible & np.asarray(
        [bool(row["passes_both_gates"]) for row in rows], dtype=bool,
    )
    if not np.array_equal(c2_gate, reported_gate):
        raise RuntimeError("recomputed fixed C2 gate differs from audited report")

    low_band_mask = np.asarray(
        [
            ("band_excess" in name or "multiresolution_mean_excess" in name)
            and any(band in name for band in ("10_30hz", "30_50hz", "50_90hz"))
            for name in b1_names
        ],
        dtype=bool,
    )
    if int(np.sum(low_band_mask)) != 27:
        raise RuntimeError("fixed 10-90 Hz B1 feature contract drift")
    representations = {
        "c2_peak13": c2,
        "b1_10_90_shape27": b1[:, low_band_mask],
        "b1_10_90_shape27_plus_c2_peak13": np.column_stack(
            (b1[:, low_band_mask], c2)
        ).astype(np.float32),
        "b1_full63_plus_c2_peak13": np.column_stack((b1, c2)).astype(np.float32),
    }
    if any(not np.isfinite(value).all() for value in representations.values()):
        raise RuntimeError("periodic representation contains non-finite values")
    return {
        "representations": representations,
        "feature_names": {
            "c2_peak13": c2_names.tolist(),
            "b1_10_90_shape27": b1_names[low_band_mask].tolist(),
            "b1_10_90_shape27_plus_c2_peak13": (
                b1_names[low_band_mask].tolist() + c2_names.tolist()
            ),
            "b1_full63_plus_c2_peak13": b1_names.tolist() + c2_names.tolist(),
        },
        "c2_gate": c2_gate,
        "c2_eligible": eligible,
        "slow_method_version": slow_method,
        "c2_method_version": str(payload.get("method_version", "")),
    }


class HorizontalExpertCache:
    """Source-only Ridge fits with train-derived margin normalization."""

    def __init__(
        self,
        *,
        representations: Mapping[str, np.ndarray],
        group: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.representations = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in representations.items()
        }
        self.group = np.asarray(group).astype(str)
        self.target = np.asarray(target, dtype=np.int64)
        self.groups = tuple(sorted(np.unique(self.group).tolist()))
        self.evidence: dict[
            tuple[tuple[str, ...], str, str, float], ExpertEvidence
        ] = {}
        self.fit_audit: dict[tuple[tuple[str, ...], str, float], dict[str, Any]] = {}

    def get(
        self,
        fit_groups: tuple[str, ...],
        predict_group: str,
        representation: str,
        alpha: float,
    ) -> ExpertEvidence:
        fit_groups = tuple(sorted(str(item) for item in fit_groups))
        key = (fit_groups, str(predict_group), str(representation), float(alpha))
        if str(predict_group) in fit_groups:
            raise RuntimeError("horizontal prediction group overlaps fit groups")
        if key not in self.evidence:
            self._compute(fit_groups, str(representation), float(alpha))
        return self.evidence[key]

    def _compute(
        self, fit_groups: tuple[str, ...], representation: str, alpha: float,
    ) -> None:
        audit_key = (fit_groups, representation, alpha)
        if audit_key in self.fit_audit:
            return
        if representation not in self.representations:
            raise KeyError(representation)
        fit = np.flatnonzero(
            np.isin(self.group, fit_groups) & np.isin(self.target, HORIZONTAL_LABELS)
        )
        excluded_groups = tuple(name for name in self.groups if name not in fit_groups)
        predict = np.flatnonzero(np.isin(self.group, excluded_groups))
        if len(fit_groups) not in (6, 7) or len(np.unique(self.target[fit])) != 2:
            raise RuntimeError("horizontal expert requires six or seven source groups")
        feature = self.representations[representation]
        scaler = StandardScaler()
        fit_feature = scaler.fit_transform(feature[fit])
        predict_feature = scaler.transform(feature[predict])
        model = RidgeClassifier(alpha=alpha, class_weight="balanced")
        model.fit(fit_feature, self.target[fit])
        train_decision = np.asarray(
            model.decision_function(fit_feature), dtype=np.float64,
        ).reshape(-1)
        margin_scale = float(max(np.median(np.abs(train_decision)), 1.0e-6))
        decision = np.asarray(
            model.decision_function(predict_feature), dtype=np.float64,
        ).reshape(-1)
        prediction = np.asarray(model.predict(predict_feature), dtype=np.int64)
        reliability = np.abs(decision) / margin_scale
        complete = ExpertEvidence(
            prediction=prediction,
            decision=decision.astype(np.float32),
            reliability=reliability.astype(np.float32),
            training_margin_scale=margin_scale,
        )
        complete.validate(len(predict))
        for predicted_group in excluded_groups:
            local = np.flatnonzero(self.group[predict] == predicted_group)
            local_evidence = ExpertEvidence(
                prediction=complete.prediction[local].copy(),
                decision=complete.decision[local].copy(),
                reliability=complete.reliability[local].copy(),
                training_margin_scale=margin_scale,
            )
            local_evidence.validate(len(local))
            self.evidence[
                (fit_groups, predicted_group, representation, alpha)
            ] = local_evidence
        self.fit_audit[audit_key] = {
            "fit_groups": list(fit_groups),
            "predicted_groups": list(excluded_groups),
            "fit_horizontal_recordings": int(len(fit)),
            "representation": representation,
            "dimensions": int(feature.shape[1]),
            "alpha": alpha,
            "training_margin_scale": margin_scale,
            "all_predicted_groups_excluded_from_fit": True,
        }


def select_horizontal_candidate(
    source_target: np.ndarray,
    candidate_prediction: Mapping[tuple[str, float], np.ndarray],
    representation_dimensions: Mapping[str, int],
) -> tuple[tuple[str, float], list[dict[str, Any]]]:
    """Select representation/alpha only by source horizontal balanced accuracy."""

    rows: list[dict[str, Any]] = []
    for (representation, alpha), prediction in candidate_prediction.items():
        metrics = _horizontal_metrics(source_target, prediction)
        rows.append(
            {
                "representation": representation,
                "dimensions": int(representation_dimensions[representation]),
                "alpha": float(alpha),
                **metrics,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["balanced_accuracy"],
            -row["macro_f1"],
            row["dimensions"],
            -row["alpha"],
            row["representation"],
        ),
    )
    selected = ranked[0]
    return (str(selected["representation"]), float(selected["alpha"])), rows


def apply_horizontal_override(
    base_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    c2_gate: np.ndarray,
    *,
    reliability: np.ndarray | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Executable gate: base says L/C, fixed C2 gate, optional source norm margin."""

    base = np.asarray(base_prediction, dtype=np.int64)
    expert = np.asarray(expert_prediction, dtype=np.int64)
    gate = np.asarray(c2_gate, dtype=bool)
    if base.shape != expert.shape or base.shape != gate.shape:
        raise ValueError("override inputs do not align")
    allowed = np.isin(base, HORIZONTAL_LABELS) & gate
    if reliability is not None:
        value = np.asarray(reliability, dtype=np.float64)
        if value.shape != base.shape or np.isnan(value).any():
            raise ValueError("invalid expert reliability")
        allowed &= value >= float(threshold)
    output = base.copy()
    output[allowed] = expert[allowed]
    return output, allowed & (output != base)


def select_h1_switch(
    source_target: np.ndarray,
    base_prediction: np.ndarray,
    override_prediction: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    scores = {
        "off": _five_metrics(source_target, base_prediction),
        "on": _five_metrics(source_target, override_prediction),
    }
    gain_f1 = scores["on"]["macro_f1"] - scores["off"]["macro_f1"]
    gain_acc = scores["on"]["accuracy"] - scores["off"]["accuracy"]
    enabled = bool(gain_f1 > 1.0e-12 or (abs(gain_f1) <= 1.0e-12 and gain_acc > 1.0e-12))
    return enabled, {"scores": scores, "selected": "on" if enabled else "off"}


def select_h2_threshold(
    source_target: np.ndarray,
    candidate_prediction: Mapping[float, np.ndarray],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    scored: list[tuple[float, dict[str, Any]]] = []
    for threshold, prediction in candidate_prediction.items():
        metrics = _five_metrics(source_target, prediction)
        scored.append((float(threshold), metrics))
        rows.append({
            "threshold": "off" if np.isinf(threshold) else float(threshold),
            **metrics,
        })
    ranked = sorted(
        scored,
        key=lambda item: (
            -item[1]["macro_f1"],
            -item[1]["accuracy"],
            -item[0],
        ),
    )
    return float(ranked[0][0]), rows


def _source_evidence_for_fold(
    *,
    held_group: str,
    groups: tuple[str, ...],
    group: np.ndarray,
    target: np.ndarray,
    c2_gate: np.ndarray,
    representations: Mapping[str, np.ndarray],
    base_cache: MultiViewBaseCache,
    expert_cache: HorizontalExpertCache,
) -> dict[str, Any]:
    """Build inner-LOMO source evidence, excluding outer and predicted source."""

    source_groups = tuple(name for name in groups if name != held_group)
    source_indices = np.concatenate(
        [np.flatnonzero(group == name) for name in source_groups]
    )
    base_parts: list[np.ndarray] = []
    expert_parts: dict[tuple[str, float], list[np.ndarray]] = {
        (representation, alpha): []
        for representation in representations
        for alpha in RIDGE_ALPHAS
    }
    reliability_parts: dict[tuple[str, float], list[np.ndarray]] = {
        key: [] for key in expert_parts
    }
    fit_sets: dict[str, list[str]] = {}
    for source_group in source_groups:
        fit_groups = tuple(
            name for name in groups if name not in (held_group, source_group)
        )
        if len(fit_groups) != 6:
            raise RuntimeError("source meta fit must contain exactly six groups")
        fit_sets[source_group] = list(fit_groups)
        base_evidence = base_cache.get(fit_groups, source_group)
        base_prediction, _, _ = plurality_condition_tie(base_evidence)
        base_parts.append(base_prediction)
        for key in expert_parts:
            representation, alpha = key
            evidence = expert_cache.get(
                fit_groups, source_group, representation, alpha,
            )
            expert_parts[key].append(evidence.prediction)
            reliability_parts[key].append(evidence.reliability)
    base_prediction = np.concatenate(base_parts)
    expert_prediction = {
        key: np.concatenate(value) for key, value in expert_parts.items()
    }
    expert_reliability = {
        key: np.concatenate(value) for key, value in reliability_parts.items()
    }
    if (
        base_prediction.shape != source_indices.shape
        or any(value.shape != source_indices.shape for value in expert_prediction.values())
    ):
        raise RuntimeError("source evidence assembly failed")
    return {
        "source_groups": source_groups,
        "source_indices": source_indices,
        "source_target": target[source_indices],
        "source_c2_gate": c2_gate[source_indices],
        "base_prediction": base_prediction,
        "expert_prediction": expert_prediction,
        "expert_reliability": expert_reliability,
        "fit_sets": fit_sets,
    }


def build_outer_fold_unscored(
    *,
    held_group: str,
    groups: tuple[str, ...],
    group: np.ndarray,
    target: np.ndarray,
    c2_gate: np.ndarray,
    representations: Mapping[str, np.ndarray],
    base_cache: MultiViewBaseCache,
    expert_cache: HorizontalExpertCache,
) -> dict[str, Any]:
    """Build one target prediction; target labels are never indexed here."""

    source = _source_evidence_for_fold(
        held_group=held_group,
        groups=groups,
        group=group,
        target=target,
        c2_gate=c2_gate,
        representations=representations,
        base_cache=base_cache,
        expert_cache=expert_cache,
    )
    dimensions = {name: value.shape[1] for name, value in representations.items()}
    selected, horizontal_selection = select_horizontal_candidate(
        source["source_target"], source["expert_prediction"], dimensions,
    )
    source_expert = source["expert_prediction"][selected]
    source_reliability = source["expert_reliability"][selected]
    source_h1, _ = apply_horizontal_override(
        source["base_prediction"], source_expert, source["source_c2_gate"],
    )
    h1_enabled, h1_selection = select_h1_switch(
        source["source_target"], source["base_prediction"], source_h1,
    )
    h2_candidates: dict[float, np.ndarray] = {}
    for threshold in H2_THRESHOLDS:
        candidate, _ = apply_horizontal_override(
            source["base_prediction"],
            source_expert,
            source["source_c2_gate"],
            reliability=source_reliability,
            threshold=threshold,
        )
        h2_candidates[threshold] = candidate
    h2_threshold, h2_selection = select_h2_threshold(
        source["source_target"], h2_candidates,
    )
    source_h1_prediction = source_h1 if h1_enabled else source["base_prediction"].copy()
    source_h2_prediction = h2_candidates[h2_threshold]

    source_groups = source["source_groups"]
    target_indices = np.flatnonzero(group == held_group)
    target_base_evidence = base_cache.get(source_groups, held_group)
    target_base, _, _ = plurality_condition_tie(target_base_evidence)
    selected_representation, selected_alpha = selected
    target_expert_evidence = expert_cache.get(
        source_groups,
        held_group,
        selected_representation,
        selected_alpha,
    )
    target_h1_on, h1_changed = apply_horizontal_override(
        target_base,
        target_expert_evidence.prediction,
        c2_gate[target_indices],
    )
    target_h1 = target_h1_on if h1_enabled else target_base.copy()
    if not h1_enabled:
        h1_changed[:] = False
    target_h2, h2_changed = apply_horizontal_override(
        target_base,
        target_expert_evidence.prediction,
        c2_gate[target_indices],
        reliability=target_expert_evidence.reliability,
        threshold=h2_threshold,
    )
    return {
        "held_group": held_group,
        "source_groups": list(source_groups),
        "source_indices": source["source_indices"],
        "source_base_prediction": source["base_prediction"],
        "source_h0_prediction": source_expert,
        "source_h0_reliability": source_reliability,
        "source_h1_prediction": source_h1_prediction,
        "source_h2_prediction": source_h2_prediction,
        "source_meta_fit_sets": source["fit_sets"],
        "target_indices": target_indices,
        "target_base_prediction": target_base,
        "target_h0_prediction": target_expert_evidence.prediction,
        "target_h0_decision": target_expert_evidence.decision,
        "target_h0_reliability": target_expert_evidence.reliability,
        "target_h1_prediction": target_h1,
        "target_h2_prediction": target_h2,
        "target_h1_changed": h1_changed,
        "target_h2_changed": h2_changed,
        "selected_representation": selected_representation,
        "selected_alpha": selected_alpha,
        "source_horizontal_selection": horizontal_selection,
        "h1_selection": h1_selection,
        "h2_selected_threshold": h2_threshold,
        "h2_selection": h2_selection,
        "outer_target_labels_read_for_fit_or_selection": False,
    }


def generate_outer_predictions_unscored(
    *,
    event: np.ndarray,
    group: np.ndarray,
    target: np.ndarray,
    c2_gate: np.ndarray,
    representations: Mapping[str, np.ndarray],
    base_cache: MultiViewBaseCache,
    expert_cache: HorizontalExpertCache,
) -> dict[str, Any]:
    groups = _validate_inventory(event, group, target)
    output = {
        name: np.full(len(target), -1, dtype=np.int64)
        for name in ("base", "H0", "H1", "H2")
    }
    decision = np.full(len(target), np.nan, dtype=np.float32)
    reliability = np.full(len(target), np.nan, dtype=np.float32)
    changed = {
        "H1": np.zeros(len(target), dtype=bool),
        "H2": np.zeros(len(target), dtype=bool),
    }
    config = {
        "selected_representation": np.full(len(target), "", dtype="<U64"),
        "selected_alpha": np.full(len(target), np.nan, dtype=np.float64),
        "H1_enabled": np.zeros(len(target), dtype=bool),
        "H2_threshold": np.full(len(target), np.nan, dtype=np.float64),
    }
    meta = {
        name: np.full((len(groups), len(target)), -1, dtype=np.int64)
        for name in ("base", "H0", "H1", "H2")
    }
    meta_reliability = np.full((len(groups), len(target)), np.nan, dtype=np.float32)
    meta_source_mask = np.zeros((len(groups), len(target)), dtype=bool)
    folds: list[dict[str, Any]] = []
    for fold_index, held_group in enumerate(groups):
        fold = build_outer_fold_unscored(
            held_group=held_group,
            groups=groups,
            group=group,
            target=target,
            c2_gate=c2_gate,
            representations=representations,
            base_cache=base_cache,
            expert_cache=expert_cache,
        )
        indices = fold["target_indices"]
        if np.any(output["base"][indices] >= 0):
            raise RuntimeError("outer target rows overlap")
        output["base"][indices] = fold["target_base_prediction"]
        output["H0"][indices] = fold["target_h0_prediction"]
        output["H1"][indices] = fold["target_h1_prediction"]
        output["H2"][indices] = fold["target_h2_prediction"]
        decision[indices] = fold["target_h0_decision"]
        reliability[indices] = fold["target_h0_reliability"]
        changed["H1"][indices] = fold["target_h1_changed"]
        changed["H2"][indices] = fold["target_h2_changed"]
        config["selected_representation"][indices] = fold["selected_representation"]
        config["selected_alpha"][indices] = fold["selected_alpha"]
        config["H1_enabled"][indices] = fold["h1_selection"]["selected"] == "on"
        config["H2_threshold"][indices] = fold["h2_selected_threshold"]
        source_indices = fold["source_indices"]
        meta_source_mask[fold_index, source_indices] = True
        meta["base"][fold_index, source_indices] = fold["source_base_prediction"]
        meta["H0"][fold_index, source_indices] = fold["source_h0_prediction"]
        meta["H1"][fold_index, source_indices] = fold["source_h1_prediction"]
        meta["H2"][fold_index, source_indices] = fold["source_h2_prediction"]
        meta_reliability[fold_index, source_indices] = fold[
            "source_h0_reliability"
        ]
        folds.append(
            {
                key: (
                    "off" if key == "h2_selected_threshold" and np.isinf(value)
                    else value
                )
                for key, value in fold.items()
                if key
                not in {
                    "target_indices",
                    "target_base_prediction",
                    "target_h0_prediction",
                    "target_h0_decision",
                    "target_h0_reliability",
                    "target_h1_prediction",
                    "target_h2_prediction",
                    "target_h1_changed",
                    "target_h2_changed",
                    "source_indices",
                    "source_base_prediction",
                    "source_h0_prediction",
                    "source_h0_reliability",
                    "source_h1_prediction",
                    "source_h2_prediction",
                }
            }
        )
        print(
            f"horizontal fold={fold_index + 1}/8 held={held_group} "
            f"expert={fold['selected_representation']} a={fold['selected_alpha']:g} "
            f"H1={fold['h1_selection']['selected']} "
            f"H2={'off' if np.isinf(fold['h2_selected_threshold']) else fold['h2_selected_threshold']:} "
            "outer_target_unscored",
            flush=True,
        )
    if (
        any(np.any(value < 0) for value in output.values())
        or not np.isfinite(decision).all()
        or not np.isfinite(reliability).all()
    ):
        raise RuntimeError("not all outer predictions were frozen")
    return {
        **output,
        "H0_decision": decision,
        "H0_reliability": reliability,
        "H1_changed": changed["H1"],
        "H2_changed": changed["H2"],
        **config,
        "meta_outer_held_groups": np.asarray(groups),
        "meta_source_mask": meta_source_mask,
        "meta_base_prediction": meta["base"],
        "meta_H0_prediction": meta["H0"],
        "meta_H0_reliability": meta_reliability,
        "meta_H1_prediction": meta["H1"],
        "meta_H2_prediction": meta["H2"],
        "folds": folds,
        "all_outer_predictions_frozen": True,
    }
