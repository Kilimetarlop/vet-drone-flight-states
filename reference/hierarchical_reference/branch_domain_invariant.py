"""Domain-invariant branch: stable feature selection, removal of the source-product-mean
subspace, an RBF SVM, and the nested choice of the penalty on held-out source products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.svm import SVC


CLASS_NAMES = (
    "LINEAR_CRUISE",
    "CURVED_TURN",
    "HOVER",
    "TAKEOFF_ASCEND",
    "DESCEND_LAND",
)


EXPECTED_EVENTS = 168


EXPECTED_GROUPS = 8


EXPECTED_COUNTS = (48, 48, 24, 24, 24)


EPSILON = 1.0e-8


@dataclass(frozen=True, order=True)
class Candidate:
    """A complete, nested-selected source-only configuration."""

    view: str
    features: int
    nuisance_rank: int
    group_penalty: float
    classifier: str

    @property
    def identifier(self) -> str:
        penalty = str(self.group_penalty).replace(".", "p")
        return (
            f"{self.view}_k{self.features}_r{self.nuisance_rank}"
            f"_gp{penalty}_{self.classifier}"
        )


def candidate_grid(
    views: Iterable[str] = ("fusion",),
    quick: bool = False,
) -> tuple[Candidate, ...]:
    """Return the deterministic search space used inside every outer fold."""
    if quick:
        sizes = (32, 128)
        ranks = (0, 2)
        penalties = (0.5,)
        classifiers = ("linear_c0.1", "rbf_c1")
    else:
        sizes = (192,)
        ranks = (4,)
        penalties = (0.25, 0.5, 1.0)
        classifiers = ("rbf_c10",)
    return tuple(
        Candidate(view, size, rank, penalty, classifier)
        for view in views
        for size in sizes
        for rank in ranks
        for penalty in penalties
        for classifier in classifiers
    )


def _load_inputs(
    feature_path: Path,
    metadata_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as feature_cache:
        missing = {"event", "global", "stft"} - set(feature_cache.files)
        if missing:
            raise RuntimeError(f"feature cache missing keys: {sorted(missing)}")
        event = feature_cache["event"].astype(str)
        global_feature = np.asarray(feature_cache["global"], dtype=np.float32)
        stft_feature = np.asarray(feature_cache["stft"], dtype=np.float32)
    with np.load(metadata_path, allow_pickle=False) as metadata_cache:
        required = {"event", "identity", "class_index"}
        missing = required - set(metadata_cache.files)
        if missing:
            raise RuntimeError(f"metadata cache missing keys: {sorted(missing)}")
        metadata_event = metadata_cache["event"].astype(str)
        group = metadata_cache["identity"].astype(str)
        target = np.asarray(metadata_cache["class_index"], dtype=np.int64)
    if not np.array_equal(event, metadata_event):
        raise RuntimeError("feature and metadata event order differs")
    if event.shape != (EXPECTED_EVENTS,) or len(set(event.tolist())) != EXPECTED_EVENTS:
        raise RuntimeError(f"expected {EXPECTED_EVENTS} unique events")
    if len(set(group.tolist())) != EXPECTED_GROUPS:
        raise RuntimeError(f"expected {EXPECTED_GROUPS} UAV/model groups")
    counts = tuple(int(np.sum(target == index)) for index in range(len(CLASS_NAMES)))
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"class-count drift: {counts}")
    for name, value in (("global", global_feature), ("stft", stft_feature)):
        if value.ndim != 2 or value.shape[0] != EXPECTED_EVENTS:
            raise RuntimeError(f"invalid {name} feature shape: {value.shape}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{name} features contain non-finite values")
    feature = {
        "global": global_feature,
        "stft": stft_feature,
        "fusion": np.concatenate((global_feature, stft_feature), axis=1),
    }
    return feature, event, group, target


def stable_contrast_scores(
    feature: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    group_penalty: float = 0.5,
) -> np.ndarray:
    """Score source features by repeatable class contrast across domains.

    Each group contains all five classes.  Removing each group's mean before
    estimating class contrasts prevents a large radio/model offset from being
    mistaken for behavior.  The denominator penalises within-class noise,
    group-by-class interaction, and residual group offsets.
    """
    value = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target)
    group = np.asarray(group)
    groups = np.unique(group)
    classes = np.unique(target)
    if value.ndim != 2 or len(value) != len(target) or len(value) != len(group):
        raise ValueError("feature, target, and group dimensions disagree")
    if len(groups) < 2 or len(classes) < 2:
        raise ValueError("stable scoring needs at least two groups and classes")

    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale[scale < 1.0e-7] = 1.0
    z = (value - mean) / scale
    group_mean = np.stack([z[group == item].mean(axis=0) for item in groups])
    contrast = np.empty((len(groups), len(classes), z.shape[1]), dtype=np.float64)
    within_sum = np.zeros(z.shape[1], dtype=np.float64)
    within_count = 0
    for group_index, group_name in enumerate(groups):
        group_mask = group == group_name
        for class_index, class_name in enumerate(classes):
            cell = z[group_mask & (target == class_name)]
            if len(cell) == 0:
                raise ValueError("every source group must contain every class")
            cell_mean = cell.mean(axis=0)
            contrast[group_index, class_index] = cell_mean - group_mean[group_index]
            within_sum += np.square(cell - cell_mean).sum(axis=0)
            within_count += len(cell)

    prototype = contrast.mean(axis=0)
    class_signal = np.mean(np.square(prototype), axis=0)
    interaction = np.mean(np.square(contrast - prototype[None]), axis=(0, 1))
    group_effect = np.mean(np.square(group_mean - group_mean.mean(axis=0)), axis=0)
    within_noise = within_sum / max(within_count, 1)
    denominator = interaction + 0.25 * within_noise + group_penalty * group_effect
    score = class_signal / np.maximum(denominator, EPSILON)
    score[~np.isfinite(score)] = -np.inf
    return score


class StableNuisanceMap:
    """Train-source feature selector and low-rank domain-removal map."""

    def __init__(
        self,
        features: int,
        nuisance_rank: int,
        group_penalty: float,
    ) -> None:
        self.features = int(features)
        self.nuisance_rank = int(nuisance_rank)
        self.group_penalty = float(group_penalty)

    def fit(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        group: np.ndarray,
    ) -> "StableNuisanceMap":
        score = stable_contrast_scores(
            feature, target, group, self.group_penalty,
        )
        count = min(self.features, feature.shape[1])
        # Stable tie-break by original feature index.
        order = np.lexsort((np.arange(len(score)), -score))
        self.selected_ = np.sort(order[:count])
        selected = np.asarray(feature[:, self.selected_], dtype=np.float64)
        self.mean_ = selected.mean(axis=0)
        self.scale_ = selected.std(axis=0)
        self.scale_[self.scale_ < 1.0e-7] = 1.0
        z = (selected - self.mean_) / self.scale_

        names = np.unique(group)
        group_mean = np.stack([z[group == name].mean(axis=0) for name in names])
        group_mean -= group_mean.mean(axis=0, keepdims=True)
        maximum_rank = min(len(names) - 1, group_mean.shape[1])
        rank = min(self.nuisance_rank, maximum_rank)
        if rank:
            _, singular, right = np.linalg.svd(group_mean, full_matrices=False)
            nonzero = int(np.sum(singular > 1.0e-8))
            rank = min(rank, nonzero)
            self.nuisance_basis_ = right[:rank].copy()
        else:
            self.nuisance_basis_ = np.empty((0, count), dtype=np.float64)
        self.effective_rank_ = int(len(self.nuisance_basis_))
        self.fit_groups_ = tuple(sorted(np.unique(group).astype(str).tolist()))
        return self

    def transform(self, feature: np.ndarray) -> np.ndarray:
        selected = np.asarray(feature[:, self.selected_], dtype=np.float64)
        z = (selected - self.mean_) / self.scale_
        if len(self.nuisance_basis_):
            z -= (z @ self.nuisance_basis_.T) @ self.nuisance_basis_
        if not np.isfinite(z).all():
            raise RuntimeError("domain-invariant transform produced non-finite values")
        return z.astype(np.float32)

    def fit_transform(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        group: np.ndarray,
    ) -> np.ndarray:
        return self.fit(feature, target, group).transform(feature)

    def state_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "nuisance_rank": self.nuisance_rank,
            "effective_rank": self.effective_rank_,
            "group_penalty": self.group_penalty,
            "selected": self.selected_,
            "mean": self.mean_,
            "scale": self.scale_,
            "nuisance_basis": self.nuisance_basis_,
            "fit_groups": self.fit_groups_,
        }


def _classifier(name: str, seed: int) -> BaseEstimator:
    if name.startswith("linear_c"):
        c_value = float(name.removeprefix("linear_c"))
        return SVC(
            C=c_value,
            kernel="linear",
            class_weight="balanced",
            decision_function_shape="ovr",
            probability=True,
            random_state=seed,
        )
    if name.startswith("rbf_c"):
        c_value = float(name.removeprefix("rbf_c"))
        return SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            decision_function_shape="ovr",
            probability=True,
            random_state=seed,
        )
    if name == "lda_shrink":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"unknown classifier: {name}")


def _fit_candidate(
    candidate: Candidate,
    features: dict[str, np.ndarray],
    target: np.ndarray,
    group: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, StableNuisanceMap, BaseEstimator]:
    source = features[candidate.view]
    mapper = StableNuisanceMap(
        candidate.features,
        candidate.nuisance_rank,
        candidate.group_penalty,
    )
    fit_feature = mapper.fit_transform(
        source[fit_indices], target[fit_indices], group[fit_indices],
    )
    predict_feature = mapper.transform(source[predict_indices])
    model = _classifier(candidate.classifier, seed)
    model.fit(fit_feature, target[fit_indices])
    prediction = np.asarray(model.predict(predict_feature), dtype=np.int64)
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(predict_feature), dtype=np.float32)
    else:  # pragma: no cover - all current candidates expose probabilities
        probability = np.eye(len(CLASS_NAMES), dtype=np.float32)[prediction]
    return prediction, probability, mapper, model


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        f1_score(
            target,
            prediction,
            labels=np.arange(len(CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
    )


def nested_select(
    candidates: tuple[Candidate, ...],
    features: dict[str, np.ndarray],
    target: np.ndarray,
    group: np.ndarray,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[Candidate, dict[str, float]]:
    """Select every method hyper-parameter using source groups only."""
    source_groups = sorted(np.unique(group[outer_train]).astype(str).tolist())
    score: dict[str, float] = {}
    for candidate_index, candidate in enumerate(candidates):
        prediction = np.full(len(outer_train), -1, dtype=np.int64)
        for inner_group_index, validation_group in enumerate(source_groups):
            validation = group[outer_train] == validation_group
            fit_local = np.flatnonzero(~validation)
            validation_local = np.flatnonzero(validation)
            fold_prediction, _, mapper, _ = _fit_candidate(
                candidate,
                {name: value[outer_train] for name, value in features.items()},
                target[outer_train],
                group[outer_train],
                fit_local,
                validation_local,
                seed + candidate_index * 101 + inner_group_index,
            )
            if validation_group in mapper.fit_groups_:
                raise RuntimeError("inner validation group leaked into mapper fit")
            prediction[validation_local] = fold_prediction
        if np.any(prediction < 0):
            raise RuntimeError("inner out-of-group predictions are incomplete")
        score[candidate.identifier] = _macro_f1(target[outer_train], prediction)
    best = min(
        candidates,
        key=lambda item: (-score[item.identifier], item.identifier),
    )
    return best, score
