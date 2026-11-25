"""Branch evidence for every fit set and the plurality vote of the four branches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping
import numpy as np
from sklearn.svm import SVC

from .branch_condition_prototype import build_robust36_features, fit_condition_prototypes, load_dense_scalar_cache, predict_condition_prototypes
from .spectral_candidates import SourceStandardMap, _stable_candidates as envelope_stable_candidates, _standard_candidates as envelope_standard_candidates, nested_select as select_spectral_candidate
from .branch_multirocket_hydra import _fit_transform_fold as fit_multirocket_hydra, align_all_records, apply_source_robust_scaler, fit_source_robust_scaler, load_cache as load_multirocket_cache
from .spectral_activity_features import load_or_build_feature_cache as load_spectral_features
from .branch_domain_invariant import CLASS_NAMES, StableNuisanceMap, _fit_candidate as fit_domain_candidate, _load_inputs as load_domain_inputs, candidate_grid as domain_candidate_grid, nested_select as select_domain_candidate
from .stacking import condition_evidence


EXPECTED_EVENTS = 168


EXPECTED_GROUPS = 8


EXPECTED_COUNTS = (48, 48, 24, 24, 24)


BASE_NAMES = (
    "DomainInvariant",
    "ConditionPrototype",
    "SpectralNested",
    "MultiRocketHydra",
)


EPSILON = 1.0e-8


@dataclass(frozen=True)
class Evidence:
    prediction: np.ndarray
    probability: np.ndarray

    def validate(self, rows: int) -> None:
        if self.prediction.shape != (rows,):
            raise RuntimeError(f"invalid prediction shape: {self.prediction.shape}")
        if self.probability.shape != (rows, len(CLASS_NAMES)):
            raise RuntimeError(f"invalid probability shape: {self.probability.shape}")
        if np.any((self.prediction < 0) | (self.prediction >= len(CLASS_NAMES))):
            raise RuntimeError("base prediction outside five-class range")
        if not np.isfinite(self.probability).all():
            raise RuntimeError("base probability contains non-finite values")
        if not np.allclose(self.probability.sum(axis=1), 1.0, atol=1.0e-5):
            raise RuntimeError("base probabilities do not sum to one")


def _softmax(value: np.ndarray) -> np.ndarray:
    score = np.asarray(value, dtype=np.float64)
    shifted = score - score.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -80.0, 0.0))
    return (
        exponential / np.maximum(exponential.sum(axis=1, keepdims=True), EPSILON)
    ).astype(np.float32)


def decision_to_probability(decision: np.ndarray) -> np.ndarray:
    """Convert incomparable branch decisions to recording-local evidence.

    Row standardisation removes arbitrary branch scale without consulting any
    other target recording.
    """
    value = np.asarray(decision, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"invalid decision matrix: {value.shape}")
    center = value.mean(axis=1, keepdims=True)
    scale = value.std(axis=1, keepdims=True)
    return _softmax((value - center) / np.maximum(scale, 1.0e-6))


def _normalise_probability(probability: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(probability, dtype=np.float64), 0.0)
    value /= np.maximum(value.sum(axis=1, keepdims=True), EPSILON)
    return value.astype(np.float32)


def plurality_condition_tie(
    evidence: Mapping[str, Evidence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four-view plurality with ConditionPrototype as a genuine tie breaker."""
    stacked = np.stack([evidence[name].prediction for name in BASE_NAMES], axis=1)
    output = np.empty(len(stacked), dtype=np.int64)
    winning_votes = np.empty(len(stacked), dtype=np.int64)
    tied = np.empty(len(stacked), dtype=bool)
    condition_index = BASE_NAMES.index("ConditionPrototype")
    for row_index, row in enumerate(stacked):
        counts = np.bincount(row, minlength=len(CLASS_NAMES))
        winners = np.flatnonzero(counts == counts.max())
        fallback = int(row[condition_index])
        winning_votes[row_index] = int(counts.max())
        tied[row_index] = len(winners) > 1
        output[row_index] = (
            fallback if fallback in winners else int(winners[0])
        )
    return output, winning_votes, tied


def _seed_for_groups(seed: int, fit_groups: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(fit_groups).encode("utf-8")).digest()
    return int((seed + int.from_bytes(digest[:4], "little")) % (2**31 - 1))


def _fit_spectral_selected(
    candidate: Any,
    views: dict[str, np.ndarray],
    target: np.ndarray,
    group: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
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
        raise ValueError(f"unknown spectral mapper: {candidate.mapper}")
    gamma: str | float = candidate.gamma
    if gamma != "scale":
        gamma = float(gamma)
    model = SVC(
        C=candidate.c_value,
        kernel=candidate.kernel,
        gamma=gamma,
        class_weight="balanced",
        decision_function_shape="ovr",
    )
    model.fit(mapper.transform(source[fit_indices]), target[fit_indices])
    transformed = mapper.transform(source[predict_indices])
    prediction = np.asarray(model.predict(transformed), dtype=np.int64)
    decision = np.asarray(model.decision_function(transformed), dtype=np.float32)
    return prediction, decision_to_probability(decision), mapper


class _QuietMonitor:
    def set_stage(self, _stage: str) -> None:
        return None


class MultiViewBaseCache:
    """Fit a source set once and predict every domain excluded from that set."""

    def __init__(
        self,
        *,
        domain_views: dict[str, np.ndarray],
        dense_feature: np.ndarray,
        condition: np.ndarray,
        spectral_views: dict[str, np.ndarray],
        aligned_trajectory: np.ndarray,
        event: np.ndarray,
        group: np.ndarray,
        target: np.ndarray,
        seed: int,
    ) -> None:
        self.domain_views = domain_views
        self.dense_feature = dense_feature
        self.condition = condition
        self.spectral_views = spectral_views
        self.aligned_trajectory = aligned_trajectory
        self.event = event
        self.group = group
        self.target = target
        self.seed = int(seed)
        self.groups = tuple(sorted(np.unique(group).astype(str).tolist()))
        self.domain_candidates = domain_candidate_grid(("fusion",), quick=False)
        dimension = spectral_views["raw"].shape[1]
        self.spectral_candidates = (
            envelope_standard_candidates("raw")
            + envelope_standard_candidates("record_robust")
            + envelope_stable_candidates(dimension)
        )
        self.evidence: dict[tuple[tuple[str, ...], str], dict[str, Evidence]] = {}
        self.fit_audit: dict[tuple[str, ...], dict[str, Any]] = {}

    def get(
        self, fit_groups: tuple[str, ...], predict_group: str,
    ) -> dict[str, Evidence]:
        fit_groups = tuple(sorted(str(item) for item in fit_groups))
        key = (fit_groups, str(predict_group))
        if predict_group in fit_groups:
            raise RuntimeError("prediction group overlaps base fit groups")
        if key not in self.evidence:
            self._compute_fit_set(fit_groups)
        return self.evidence[key]

    def _compute_fit_set(self, fit_groups: tuple[str, ...]) -> None:
        if fit_groups in self.fit_audit:
            return
        fit_indices = np.flatnonzero(np.isin(self.group, fit_groups))
        excluded_groups = tuple(name for name in self.groups if name not in fit_groups)
        predict_indices = np.flatnonzero(np.isin(self.group, excluded_groups))
        if len(fit_groups) not in (6, 7) or not len(predict_indices):
            raise RuntimeError("router only supports six- or seven-domain base fits")
        split_seed = _seed_for_groups(self.seed, fit_groups)

        domain_selected, domain_scores = select_domain_candidate(
            self.domain_candidates,
            self.domain_views,
            self.target,
            self.group,
            fit_indices,
            split_seed,
        )
        domain_prediction, domain_probability, domain_mapper, _ = fit_domain_candidate(
            domain_selected,
            self.domain_views,
            self.target,
            self.group,
            fit_indices,
            predict_indices,
            split_seed + 1,
        )
        domain_probability = _normalise_probability(domain_probability)

        condition_model = fit_condition_prototypes(
            self.dense_feature, self.condition, fit_indices,
        )
        condition_prediction, _, _ = predict_condition_prototypes(
            condition_model, self.dense_feature[predict_indices],
        )
        condition_probability = _normalise_probability(
            condition_evidence(
                condition_model, self.dense_feature[predict_indices],
            )["minimum_probability"]
        )

        spectral_selected, spectral_scores = select_spectral_candidate(
            self.spectral_candidates,
            self.spectral_views,
            self.target,
            self.group,
            fit_indices,
        )
        spectral_prediction, spectral_probability, spectral_mapper = (
            _fit_spectral_selected(
                spectral_selected,
                self.spectral_views,
                self.target,
                self.group,
                fit_indices,
                predict_indices,
            )
        )

        center, scale = fit_source_robust_scaler(
            self.aligned_trajectory[fit_indices],
        )
        mr_source = apply_source_robust_scaler(
            self.aligned_trajectory[fit_indices], center, scale,
        )
        mr_predict = apply_source_robust_scaler(
            self.aligned_trajectory[predict_indices], center, scale,
        )
        mr_prediction, mr_decision, mr_alpha, mr_audit = fit_multirocket_hydra(
            mr_source,
            mr_predict,
            self.target[fit_indices],
            _QuietMonitor(),
        )
        mr_probability = decision_to_probability(mr_decision)

        complete = {
            "DomainInvariant": Evidence(domain_prediction, domain_probability),
            "ConditionPrototype": Evidence(
                condition_prediction, condition_probability,
            ),
            "SpectralNested": Evidence(
                spectral_prediction, spectral_probability,
            ),
            "MultiRocketHydra": Evidence(mr_prediction, mr_probability),
        }
        for evidence in complete.values():
            evidence.validate(len(predict_indices))
        for predict_group in excluded_groups:
            local = np.flatnonzero(self.group[predict_indices] == predict_group)
            self.evidence[(fit_groups, predict_group)] = {
                name: Evidence(
                    branch.prediction[local].copy(),
                    branch.probability[local].copy(),
                )
                for name, branch in complete.items()
            }
        if set(domain_mapper.fit_groups_) != set(fit_groups):
            raise RuntimeError("domain mapper source-group audit failed")
        if set(spectral_mapper.fit_groups_) != set(fit_groups):
            raise RuntimeError("spectral mapper source-group audit failed")
        self.fit_audit[fit_groups] = {
            "fit_groups": list(fit_groups),
            "predicted_groups": list(excluded_groups),
            "fit_events": int(len(fit_indices)),
            "predicted_events": int(len(predict_indices)),
            "all_predicted_groups_excluded_from_fit": True,
            "domain_selected_candidate": domain_selected.identifier,
            "domain_selected_inner_macro_f1": float(
                domain_scores[domain_selected.identifier]
            ),
            "spectral_selected_candidate": spectral_selected.identifier,
            "spectral_selected_inner_macro_f1": float(
                spectral_scores[spectral_selected.identifier]
            ),
            "multirocket_hydra_alpha": float(np.asarray(mr_alpha).reshape(-1)[0]),
            "multirocket_hydra_dimensions": mr_audit["feature_dimensions"],
        }
        print(
            f"base fit groups={len(fit_groups)} excluded={','.join(excluded_groups)} "
            f"domain={domain_selected.identifier} "
            f"spectral={spectral_selected.identifier}",
            flush=True,
        )


def _load_all(args: argparse.Namespace) -> dict[str, Any]:
    domain_views, event, group, target = load_domain_inputs(
        args.domain_features, args.metadata,
    )
    dense = load_dense_scalar_cache(args.dense_cache)
    spectral = load_spectral_features(
        args.spectral_trajectory, args.spectral_features,
    )
    multirocket = load_multirocket_cache(args.spectral_trajectory)
    for name, other_event in (
        ("dense", dense["event"]),
        ("spectral", spectral["event"]),
        ("multirocket", multirocket["event"]),
    ):
        if not np.array_equal(event, np.asarray(other_event).astype(str)):
            raise RuntimeError(f"{name} event order differs")
    if not (
        np.array_equal(group, dense["identity"])
        and np.array_equal(group, spectral["identity"].astype(str))
        and np.array_equal(group, multirocket["identity"].astype(str))
    ):
        raise RuntimeError("base cache group order differs")
    if not (
        np.array_equal(target, dense["target"])
        and np.array_equal(target, spectral["target"].astype(np.int64))
        and np.array_equal(target, multirocket["target"].astype(np.int64))
    ):
        raise RuntimeError("base cache target order differs")
    counts = tuple(int(np.sum(target == index)) for index in range(len(CLASS_NAMES)))
    if (
        len(event) != EXPECTED_EVENTS
        or len(np.unique(group)) != EXPECTED_GROUPS
        or counts != EXPECTED_COUNTS
    ):
        raise RuntimeError("five-behavior inventory drift")
    aligned, alignment_audit = align_all_records(multirocket["trajectory"])
    return {
        "domain_views": domain_views,
        "dense_feature": build_robust36_features(dense["scalar"]),
        "condition": dense["condition"],
        "spectral_views": {
            "raw": spectral["raw"].astype(np.float32),
            "record_robust": spectral["record_robust"].astype(np.float32),
        },
        "aligned_trajectory": aligned,
        "alignment_audit": alignment_audit,
        "event": event,
        "group": group,
        "target": target,
    }
