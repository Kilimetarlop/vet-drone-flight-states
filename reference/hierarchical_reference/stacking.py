"""Condition distances and class-balanced probabilities of the condition-prototype branch.
"""

from __future__ import annotations

import numpy as np

from .branch_condition_prototype import ConditionPrototypeModel, condition_to_class


EPSILON = 1.0e-8


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponential / np.maximum(exponential.sum(axis=1, keepdims=True), EPSILON)


def condition_evidence(
    model: ConditionPrototypeModel,
    feature: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return class-balanced condition distances and probabilities per event."""
    model.validate()
    value = np.asarray(feature, dtype=np.float64)
    standardized = (value - model.scaler_mean) / model.scaler_scale
    distance = np.square(
        standardized[:, None, :] - model.condition_prototypes[None, :, :]
    ).mean(axis=2)
    row_center = distance.mean(axis=1, keepdims=True)
    row_scale = distance.std(axis=1, keepdims=True)
    relative = (distance - row_center) / np.maximum(row_scale, 1.0e-6)

    condition_class = condition_to_class(model.condition_ids)
    macro_minimum = np.stack(
        [distance[:, condition_class == class_id].min(axis=1) for class_id in range(5)],
        axis=1,
    )
    macro_relative = (
        macro_minimum - macro_minimum.mean(axis=1, keepdims=True)
    ) / np.maximum(macro_minimum.std(axis=1, keepdims=True), 1.0e-6)
    minimum_probability = _softmax(-macro_relative)

    condition_probability = _softmax(-relative)
    balanced_probability = np.stack(
        [
            condition_probability[:, condition_class == class_id].mean(axis=1)
            for class_id in range(5)
        ],
        axis=1,
    )
    balanced_probability /= np.maximum(
        balanced_probability.sum(axis=1, keepdims=True), EPSILON,
    )
    nearest_index = distance.argmin(axis=1)
    nearest_condition = model.condition_ids[nearest_index]
    nearest_class = condition_to_class(nearest_condition)
    ordered_distance = np.sort(relative, axis=1)
    condition_gap = ordered_distance[:, 1] - ordered_distance[:, 0]
    return {
        "distance": distance.astype(np.float32),
        "relative_distance": relative.astype(np.float32),
        "macro_minimum_distance": macro_minimum.astype(np.float32),
        "macro_relative_distance": macro_relative.astype(np.float32),
        "minimum_probability": minimum_probability.astype(np.float32),
        "balanced_probability": balanced_probability.astype(np.float32),
        "nearest_condition": nearest_condition.astype(np.int64),
        "nearest_class": nearest_class.astype(np.int64),
        "minimum_distance": distance[
            np.arange(len(distance)), nearest_index
        ].astype(np.float32),
        "condition_gap": condition_gap.astype(np.float32),
    }
