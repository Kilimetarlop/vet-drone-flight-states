"""Condition-prototype branch: 36 gain-robust descriptors and the nearest source
condition mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = (
    "LINEAR_CRUISE",
    "CURVED_TURN",
    "HOVER",
    "TAKEOFF_ASCEND",
    "DESCEND_LAND",
)


EXPECTED_EVENTS = 168


EXPECTED_IDENTITIES = 8


EXPECTED_CONDITIONS = np.arange(1, 22, dtype=np.int64)


EXPECTED_SCALAR_NAMES = (
    "log_rms",
    "frame_log_power_mean",
    "frame_log_power_std",
    "frame_log_power_q90_minus_q10",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
)


ROBUST_SCALAR_INDICES = (2, 3, 4, 5, 6, 7)


ROBUST_SCALAR_NAMES = tuple(
    EXPECTED_SCALAR_NAMES[index] for index in ROBUST_SCALAR_INDICES
)


SEGMENT_NAMES = ("early_0p1s", "middle_0p1s", "late_0p1s")


def _feature_names() -> tuple[str, ...]:
    raw = tuple(
        f"{segment}__{feature}"
        for segment in SEGMENT_NAMES
        for feature in ROBUST_SCALAR_NAMES
    )
    relative = tuple(
        f"{segment}__relative_to_three_segment_mean__{feature}"
        for segment in SEGMENT_NAMES
        for feature in ROBUST_SCALAR_NAMES
    )
    return raw + relative


FEATURE_NAMES = _feature_names()


def condition_to_class(condition: np.ndarray | Iterable[int]) -> np.ndarray:
    value = np.asarray(condition, dtype=np.int64)
    if np.any((value < 1) | (value > 21)):
        raise ValueError("condition indices must be in c1..c21")
    return np.searchsorted(
        np.asarray((6, 12, 15, 18, 21), dtype=np.int64),
        value,
        side="left",
    ).astype(np.int64)


def build_robust36_features(segment_features: np.ndarray) -> np.ndarray:
    value = np.asarray(segment_features, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (3, len(EXPECTED_SCALAR_NAMES)):
        raise RuntimeError(
            "segment features must have shape (recordings, 3, 8), got "
            f"{value.shape}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError("segment features contain non-finite values")
    robust = value[:, :, ROBUST_SCALAR_INDICES]
    relative = robust - robust.mean(axis=1, keepdims=True)
    result = np.concatenate(
        (robust.reshape(len(value), -1), relative.reshape(len(value), -1)),
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(value), len(FEATURE_NAMES)):
        raise RuntimeError(f"frozen feature dimension drift: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("frozen features contain non-finite values")
    return result


@dataclass(frozen=True)
class ConditionPrototypeModel:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    condition_ids: np.ndarray
    condition_prototypes: np.ndarray

    def validate(self) -> None:
        dimensions = len(FEATURE_NAMES)
        if self.scaler_mean.shape != (dimensions,):
            raise RuntimeError("invalid scaler mean")
        if self.scaler_scale.shape != (dimensions,):
            raise RuntimeError("invalid scaler scale")
        if np.any(self.scaler_scale <= 0) or not np.isfinite(self.scaler_scale).all():
            raise RuntimeError("invalid scaler scale values")
        if not np.array_equal(self.condition_ids, EXPECTED_CONDITIONS):
            raise RuntimeError("model must contain ordered c1..c21 prototypes")
        if self.condition_prototypes.shape != (21, dimensions):
            raise RuntimeError("invalid condition prototype matrix")
        if not np.isfinite(self.condition_prototypes).all():
            raise RuntimeError("condition prototypes contain non-finite values")


def fit_condition_prototypes(
    feature: np.ndarray,
    condition: np.ndarray,
    fit_indices: np.ndarray,
) -> ConditionPrototypeModel:
    """Fit a standardizer and c1..c21 prototypes from explicit source rows."""

    value = np.asarray(feature, dtype=np.float32)
    labels = np.asarray(condition, dtype=np.int64)
    indices = np.asarray(fit_indices, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"unexpected feature matrix: {value.shape}")
    if labels.shape != (len(value),):
        raise RuntimeError("condition vector does not match feature rows")
    if indices.ndim != 1 or len(indices) == 0:
        raise RuntimeError("fit indices must be a non-empty vector")
    if np.any((indices < 0) | (indices >= len(value))):
        raise RuntimeError("fit indices are out of bounds")
    source_conditions = labels[indices]
    if not np.array_equal(np.unique(source_conditions), EXPECTED_CONDITIONS):
        raise RuntimeError("source fit must contain every condition c1..c21")

    scaler = StandardScaler().fit(value[indices])
    standardized = scaler.transform(value[indices])
    prototypes = np.stack(
        [
            standardized[source_conditions == condition_id].mean(axis=0)
            for condition_id in EXPECTED_CONDITIONS
        ]
    ).astype(np.float64)
    model = ConditionPrototypeModel(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        condition_ids=EXPECTED_CONDITIONS.copy(),
        condition_prototypes=prototypes,
    )
    model.validate()
    return model


def predict_condition_prototypes(
    model: ConditionPrototypeModel,
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict macro class, nearest condition, and mean squared distance."""

    model.validate()
    value = np.asarray(feature, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"unexpected inference feature matrix: {value.shape}")
    standardized = (value - model.scaler_mean) / model.scaler_scale
    squared_distance = np.square(
        standardized[:, None, :] - model.condition_prototypes[None, :, :]
    ).mean(axis=2)
    nearest_index = squared_distance.argmin(axis=1)
    predicted_condition = model.condition_ids[nearest_index]
    predicted_class = condition_to_class(predicted_condition)
    minimum_distance = squared_distance[
        np.arange(len(squared_distance)), nearest_index
    ]
    return (
        predicted_class.astype(np.int64),
        predicted_condition.astype(np.int64),
        minimum_distance.astype(np.float64),
    )


def load_dense_scalar_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "event",
            "identity",
            "behavior",
            "class_index",
            "condition",
            "class_names",
            "segment_feature_names",
            "segment_features",
        }
        missing = sorted(required - set(cache.files))
        if missing:
            raise RuntimeError(f"dense cache missing keys: {missing}")
        result = {
            "event": cache["event"].astype(str),
            "identity": cache["identity"].astype(str),
            "behavior": cache["behavior"].astype(str),
            "target": np.asarray(cache["class_index"], dtype=np.int64),
            "condition": np.asarray(cache["condition"], dtype=np.int64),
            "class_names": cache["class_names"].astype(str),
            "scalar_names": cache["segment_feature_names"].astype(str),
            "scalar": np.asarray(cache["segment_features"], dtype=np.float32),
        }
    if len(result["event"]) != EXPECTED_EVENTS:
        raise RuntimeError(f"expected {EXPECTED_EVENTS} recordings")
    if len(set(result["event"].tolist())) != EXPECTED_EVENTS:
        raise RuntimeError("recording event identifiers must be unique")
    if len(set(result["identity"].tolist())) != EXPECTED_IDENTITIES:
        raise RuntimeError(f"expected {EXPECTED_IDENTITIES} UAV/model groups")
    if not np.array_equal(result["class_names"], np.asarray(CLASS_NAMES)):
        raise RuntimeError("five-class name order drift")
    if not np.array_equal(result["scalar_names"], np.asarray(EXPECTED_SCALAR_NAMES)):
        raise RuntimeError("dense scalar feature name drift")
    if not np.array_equal(np.unique(result["condition"]), EXPECTED_CONDITIONS):
        raise RuntimeError("dense cache must contain c1..c21")
    if not np.array_equal(
        np.bincount(result["condition"], minlength=22)[1:],
        np.full(21, EXPECTED_IDENTITIES),
    ):
        raise RuntimeError("each condition must occur once per UAV/model group")
    if not np.array_equal(result["target"], condition_to_class(result["condition"])):
        raise RuntimeError("condition and macro behavior labels disagree")
    build_robust36_features(result["scalar"])
    return result
