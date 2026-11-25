"""Candidates and nested selection of the spectral branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .branch_domain_invariant import StableNuisanceMap
from .dualtime import _metrics


BURST_THRESHOLDS = (0.5, 1.0, 2.0, 3.0)


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edge = np.diff(padded)
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    return (ends - starts).astype(np.float64)


def _burst_features(value: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median)))
    scale = max(1.4826 * mad, 1.0e-6)
    z = (value - median) / scale
    features: list[float] = []
    names: list[str] = []
    for threshold in BURST_THRESHOLDS:
        active = z > threshold
        runs = _run_lengths(active)
        gaps = _run_lengths(~active)
        if not len(runs):
            runs = np.zeros(1, dtype=np.float64)
        if not len(gaps):
            gaps = np.zeros(1, dtype=np.float64)
        denominator = float(len(value))
        values = (
            active.mean(), np.count_nonzero(np.diff(active)) / max(len(value) - 1, 1),
            len(runs) / denominator, runs.mean() / denominator,
            np.quantile(runs, 0.90) / denominator, runs.max() / denominator,
            gaps.mean() / denominator, np.quantile(gaps, 0.90) / denominator,
            gaps.max() / denominator,
        )
        labels = (
            "duty", "transition_rate", "run_count_rate", "run_mean_ratio",
            "run_q90_ratio", "run_max_ratio", "gap_mean_ratio",
            "gap_q90_ratio", "gap_max_ratio",
        )
        threshold_name = str(threshold).replace(".", "p")
        features.extend(float(item) for item in values)
        names.extend(f"burst_z{threshold_name}_{label}" for label in labels)
    for threshold in BURST_THRESHOLDS:
        threshold_name = str(threshold).replace(".", "p")
        features.append(float(np.mean(z < -threshold)))
        names.append(f"low_activity_z{threshold_name}_duty")
    return np.asarray(features, dtype=np.float64), tuple(names)


@dataclass(frozen=True, order=True)
class Candidate:
    view: str
    mapper: str
    features: int
    nuisance_rank: int
    group_penalty: float
    kernel: str
    c_value: float
    gamma: str

    @property
    def identifier(self) -> str:
        return (
            f"{self.view}_{self.mapper}_k{self.features}_r{self.nuisance_rank}_"
            f"gp{self.group_penalty:g}_{self.kernel}_c{self.c_value:g}_g{self.gamma}"
        ).replace(".", "p")


class SourceStandardMap:
    def fit(
        self, feature: np.ndarray, target: np.ndarray, group: np.ndarray,
    ) -> "SourceStandardMap":
        del target
        self.scaler_ = StandardScaler().fit(feature)
        self.fit_groups_ = tuple(sorted(np.unique(group).astype(str).tolist()))
        return self

    def transform(self, feature: np.ndarray) -> np.ndarray:
        return np.asarray(self.scaler_.transform(feature), dtype=np.float32)


def _fit_candidate(
    candidate: Candidate,
    views: dict[str, np.ndarray],
    target: np.ndarray,
    group: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> tuple[np.ndarray, Any]:
    source = views[candidate.view]
    if candidate.mapper == "standard":
        mapper: Any = SourceStandardMap().fit(
            source[fit_indices], target[fit_indices], group[fit_indices],
        )
    elif candidate.mapper == "stable_nuisance":
        mapper = StableNuisanceMap(
            candidate.features, candidate.nuisance_rank, candidate.group_penalty,
        ).fit(source[fit_indices], target[fit_indices], group[fit_indices])
    else:
        raise ValueError(f"unknown mapper: {candidate.mapper}")
    train_feature = mapper.transform(source[fit_indices])
    predict_feature = mapper.transform(source[predict_indices])
    gamma: str | float = candidate.gamma
    if gamma != "scale":
        gamma = float(gamma)
    model = SVC(
        C=candidate.c_value, kernel=candidate.kernel, gamma=gamma,
        class_weight="balanced", decision_function_shape="ovr",
    )
    model.fit(train_feature, target[fit_indices])
    return np.asarray(model.predict(predict_feature), dtype=np.int64), mapper


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(_metrics(target, prediction)["macro_f1"])


def nested_select(
    candidates: tuple[Candidate, ...],
    views: dict[str, np.ndarray],
    target: np.ndarray,
    group: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[Candidate, dict[str, float]]:
    source_groups = sorted(np.unique(group[outer_train]).astype(str).tolist())
    scores: dict[str, float] = {}
    local_views = {name: value[outer_train] for name, value in views.items()}
    local_target = target[outer_train]
    local_group = group[outer_train]
    for candidate in candidates:
        prediction = np.full(len(outer_train), -1, dtype=np.int64)
        for validation_group in source_groups:
            validation = local_group == validation_group
            fit_indices = np.flatnonzero(~validation)
            validation_indices = np.flatnonzero(validation)
            fold_prediction, mapper = _fit_candidate(
                candidate, local_views, local_target, local_group,
                fit_indices, validation_indices,
            )
            if validation_group in mapper.fit_groups_:
                raise RuntimeError("inner validation source leaked into mapper fit")
            prediction[validation_indices] = fold_prediction
        if np.any(prediction < 0):
            raise RuntimeError("inner predictions are incomplete")
        scores[candidate.identifier] = _macro_f1(local_target, prediction)
    selected = min(
        candidates, key=lambda candidate: (-scores[candidate.identifier], candidate.identifier),
    )
    return selected, scores


def _standard_candidates(view: str) -> tuple[Candidate, ...]:
    candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        candidates.append(Candidate(view, "standard", 0, 0, 0.0, "linear", c_value, "scale"))
    for c_value in (0.1, 1.0, 10.0, 100.0):
        for gamma in ("scale", "0.001", "0.01", "0.1"):
            candidates.append(Candidate(view, "standard", 0, 0, 0.0, "rbf", c_value, gamma))
    return tuple(candidates)


def _stable_candidates(feature_dimension: int) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(view, "stable_nuisance", features, rank, penalty, "rbf", c_value, "scale")
        for view in ("raw", "record_robust")
        for features in (min(192, feature_dimension), min(320, feature_dimension))
        for rank in (2, 4)
        for penalty in (0.25, 1.0)
        for c_value in (1.0, 10.0)
    )
