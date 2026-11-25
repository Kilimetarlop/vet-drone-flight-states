"""Held-out source predictions with strict double exclusion: every choice for source product g of
fold t uses only the six products left after excluding t and g.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import router as router
from .horizontal_expert import ExpertEvidence, HORIZONTAL_LABELS, HorizontalExpertCache, RIDGE_ALPHAS, _validate_inventory, apply_horizontal_override, select_h1_switch, select_horizontal_candidate
from .topology_veto import AGREEMENT_BRANCH, STRICT_BRANCH, VGAP_THRESHOLDS, apply_hover_specialist, apply_strict_veto, select_binary_switch, select_vgap_threshold, threshold_label


POISON_SENTINEL = -777


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StrictFiveSixBaseCache(router.MultiViewBaseCache):
    def _compute_fit_set(self, fit_groups: tuple[str, ...]) -> None:
        if fit_groups in self.fit_audit:
            return
        fit_indices = np.flatnonzero(np.isin(self.group, fit_groups))
        excluded_groups = tuple(name for name in self.groups if name not in fit_groups)
        predict_indices = np.flatnonzero(np.isin(self.group, excluded_groups))
        if len(fit_groups) not in (5, 6) or not len(predict_indices):
            raise RuntimeError("strict repair requires five- or six-domain base fits")
        split_seed = router._seed_for_groups(self.seed, fit_groups)

        domain_selected, domain_scores = router.select_domain_candidate(
            self.domain_candidates,
            self.domain_views,
            self.target,
            self.group,
            fit_indices,
            split_seed,
        )
        domain_prediction, domain_probability, domain_mapper, _ = (
            router.fit_domain_candidate(
                domain_selected,
                self.domain_views,
                self.target,
                self.group,
                fit_indices,
                predict_indices,
                split_seed + 1,
            )
        )
        domain_probability = router._normalise_probability(domain_probability)

        condition_model = router.fit_condition_prototypes(
            self.dense_feature, self.condition, fit_indices,
        )
        condition_prediction, _, _ = router.predict_condition_prototypes(
            condition_model, self.dense_feature[predict_indices],
        )
        condition_probability = router._normalise_probability(
            router.condition_evidence(
                condition_model, self.dense_feature[predict_indices],
            )["minimum_probability"]
        )

        spectral_selected, spectral_scores = router.select_spectral_candidate(
            self.spectral_candidates,
            self.spectral_views,
            self.target,
            self.group,
            fit_indices,
        )
        source = self.spectral_views[spectral_selected.view]
        if spectral_selected.mapper == "standard":
            spectral_mapper: Any = router.SourceStandardMap().fit(
                source[fit_indices], self.target[fit_indices], self.group[fit_indices],
            )
        elif spectral_selected.mapper == "stable_nuisance":
            spectral_mapper = router.StableNuisanceMap(
                spectral_selected.features,
                spectral_selected.nuisance_rank,
                spectral_selected.group_penalty,
            ).fit(source[fit_indices], self.target[fit_indices], self.group[fit_indices])
        else:
            raise ValueError(f"unknown spectral mapper: {spectral_selected.mapper}")
        gamma: str | float = spectral_selected.gamma
        if gamma != "scale":
            gamma = float(gamma)
        spectral_model = SVC(
            C=spectral_selected.c_value,
            kernel=spectral_selected.kernel,
            gamma=gamma,
            class_weight="balanced",
            decision_function_shape="ovr",
        )
        spectral_model.fit(
            spectral_mapper.transform(source[fit_indices]), self.target[fit_indices],
        )
        spectral_transformed = spectral_mapper.transform(source[predict_indices])
        spectral_prediction = np.asarray(
            spectral_model.predict(spectral_transformed), dtype=np.int64,
        )
        spectral_probability = router.decision_to_probability(
            np.asarray(
                spectral_model.decision_function(spectral_transformed),
                dtype=np.float32,
            )
        )

        center, scale = router.fit_source_robust_scaler(
            self.aligned_trajectory[fit_indices],
        )
        mr_source = router.apply_source_robust_scaler(
            self.aligned_trajectory[fit_indices], center, scale,
        )
        mr_predict = router.apply_source_robust_scaler(
            self.aligned_trajectory[predict_indices], center, scale,
        )
        mr_prediction, mr_decision, mr_alpha, mr_audit = (
            router.fit_multirocket_hydra(
                mr_source,
                mr_predict,
                self.target[fit_indices],
                router._QuietMonitor(),
            )
        )
        mr_probability = router.decision_to_probability(mr_decision)

        complete = {
            "DomainInvariant": router.Evidence(
                domain_prediction, domain_probability,
            ),
            "ConditionPrototype": router.Evidence(
                condition_prediction, condition_probability,
            ),
            "SpectralNested": router.Evidence(
                spectral_prediction, spectral_probability,
            ),
            "MultiRocketHydra": router.Evidence(
                mr_prediction, mr_probability,
            ),
        }
        for evidence in complete.values():
            evidence.validate(len(predict_indices))
        for predict_group in excluded_groups:
            local = np.flatnonzero(self.group[predict_indices] == predict_group)
            self.evidence[(fit_groups, predict_group)] = {
                name: router.Evidence(
                    branch.prediction[local].copy(), branch.probability[local].copy(),
                )
                for name, branch in complete.items()
            }
        if set(domain_mapper.fit_groups_) != set(fit_groups):
            raise RuntimeError("strict domain mapper fit-group audit failed")
        if set(spectral_mapper.fit_groups_) != set(fit_groups):
            raise RuntimeError("strict spectral mapper fit-group audit failed")
        self.fit_audit[fit_groups] = {
            "fit_groups": list(fit_groups),
            "predicted_groups": list(excluded_groups),
            "fit_events": int(len(fit_indices)),
            "fit_group_count": int(len(fit_groups)),
            "all_predicted_groups_excluded_from_fit": True,
            "domain_selected_candidate": domain_selected.identifier,
            "domain_selection_digest": _json_sha256(domain_scores),
            "spectral_selected_candidate": spectral_selected.identifier,
            "spectral_selection_digest": _json_sha256(spectral_scores),
            "multirocket_hydra_alpha": float(np.asarray(mr_alpha).reshape(-1)[0]),
            "multirocket_hydra_dimensions": mr_audit["feature_dimensions"],
        }
        print(
            f"strict-base fit={len(fit_groups)} excluded={','.join(excluded_groups)}",
            flush=True,
        )


class StrictFiveSixHorizontalCache(HorizontalExpertCache):
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
        if len(fit_groups) not in (5, 6) or len(np.unique(self.target[fit])) != 2:
            raise RuntimeError("strict horizontal expert requires five/six groups")
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
        complete = ExpertEvidence(
            prediction=np.asarray(model.predict(predict_feature), dtype=np.int64),
            decision=decision.astype(np.float32),
            reliability=(np.abs(decision) / margin_scale).astype(np.float32),
            training_margin_scale=margin_scale,
        )
        complete.validate(len(predict))
        for predicted_group in excluded_groups:
            local = np.flatnonzero(self.group[predict] == predicted_group)
            evidence = ExpertEvidence(
                prediction=complete.prediction[local].copy(),
                decision=complete.decision[local].copy(),
                reliability=complete.reliability[local].copy(),
                training_margin_scale=margin_scale,
            )
            evidence.validate(len(local))
            self.evidence[(fit_groups, predicted_group, representation, alpha)] = evidence
        self.fit_audit[audit_key] = {
            "fit_groups": list(fit_groups),
            "predicted_groups": list(excluded_groups),
            "fit_horizontal_recordings": int(len(fit)),
            "fit_group_count": int(len(fit_groups)),
            "representation": representation,
            "dimensions": int(feature.shape[1]),
            "alpha": float(alpha),
            "training_margin_scale": margin_scale,
            "all_predicted_groups_excluded_from_fit": True,
        }


def _make_caches(
    data: Mapping[str, Any],
    representations: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[StrictFiveSixBaseCache, StrictFiveSixHorizontalCache]:
    base = StrictFiveSixBaseCache(
        domain_views=data["domain_views"],
        dense_feature=data["dense_feature"],
        condition=data["condition"],
        spectral_views=data["spectral_views"],
        aligned_trajectory=data["aligned_trajectory"],
        event=data["event"],
        group=data["group"],
        target=data["target"],
        seed=seed,
    )
    horizontal = StrictFiveSixHorizontalCache(
        representations=representations,
        group=data["group"],
        target=data["target"],
    )
    return base, horizontal


def _select_pair_config(
    *,
    outer_h: str,
    predicted_d: str,
    groups: tuple[str, ...],
    group: np.ndarray,
    selection_target: np.ndarray,
    c2_gate: np.ndarray,
    representations: Mapping[str, np.ndarray],
    base_cache: StrictFiveSixBaseCache,
    horizontal_cache: StrictFiveSixHorizontalCache,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if outer_h == predicted_d:
        raise ValueError("h and d must differ")
    remaining = tuple(name for name in groups if name not in (outer_h, predicted_d))
    if len(remaining) != 6:
        raise RuntimeError("strict pair must leave exactly six selection groups")
    selection_indices = np.concatenate(
        [np.flatnonzero(group == name) for name in remaining]
    )
    if len(selection_indices) != 126:
        raise RuntimeError("strict pair must expose 126 selection rows")
    selection_labels = np.asarray(selection_target, dtype=np.int64)[selection_indices]
    if np.any(selection_labels == POISON_SENTINEL):
        raise RuntimeError("poisoned h/d label reached the R-only selector")

    base_parts: list[np.ndarray] = []
    strict_prediction_parts: list[np.ndarray] = []
    strict_probability_parts: list[np.ndarray] = []
    condition_parts: list[np.ndarray] = []
    expert_parts: dict[tuple[str, float], list[np.ndarray]] = {
        (representation, alpha): []
        for representation in representations
        for alpha in RIDGE_ALPHAS
    }
    fit_sets: dict[str, list[str]] = {}
    for validation_v in remaining:
        fit_groups = tuple(name for name in remaining if name != validation_v)
        if len(fit_groups) != 5 or outer_h in fit_groups or predicted_d in fit_groups:
            raise RuntimeError("five-domain inner fit-set exclusion failed")
        fit_sets[validation_v] = list(fit_groups)
        evidence = base_cache.get(fit_groups, validation_v)
        base_prediction, _, _ = router.plurality_condition_tie(evidence)
        base_parts.append(base_prediction)
        strict_prediction_parts.append(evidence[STRICT_BRANCH].prediction)
        strict_probability_parts.append(evidence[STRICT_BRANCH].probability)
        condition_parts.append(evidence[AGREEMENT_BRANCH].prediction)
        for key in expert_parts:
            representation, alpha = key
            horizontal = horizontal_cache.get(
                fit_groups, validation_v, representation, alpha,
            )
            expert_parts[key].append(horizontal.prediction)

    base_prediction = np.concatenate(base_parts)
    strict_prediction = np.concatenate(strict_prediction_parts)
    strict_probability = np.concatenate(strict_probability_parts)
    condition_prediction = np.concatenate(condition_parts)
    expert_prediction = {
        key: np.concatenate(parts) for key, parts in expert_parts.items()
    }
    if (
        base_prediction.shape != (126,)
        or strict_prediction.shape != (126,)
        or strict_probability.shape != (126, 5)
        or condition_prediction.shape != (126,)
    ):
        raise RuntimeError("six-domain selection OOF assembly failed")

    dimensions = {name: value.shape[1] for name, value in representations.items()}
    selected, horizontal_scores = select_horizontal_candidate(
        selection_labels, expert_prediction, dimensions,
    )
    selected_representation, selected_alpha = selected
    selected_expert = expert_prediction[selected]
    h1_on, _ = apply_horizontal_override(
        base_prediction, selected_expert, c2_gate[selection_indices],
    )
    h1_enabled, h1_scores = select_h1_switch(
        selection_labels, base_prediction, h1_on,
    )
    h1_prediction = h1_on if h1_enabled else base_prediction.copy()

    h3a_prediction, _ = apply_strict_veto(
        h1_prediction, strict_prediction, strict_probability,
    )
    h3b_candidates = {
        threshold: apply_strict_veto(
            h1_prediction,
            strict_prediction,
            strict_probability,
            threshold=threshold,
        )[0]
        for threshold in VGAP_THRESHOLDS
    }
    h3b_threshold, h3b_scores = select_vgap_threshold(
        selection_labels, h3b_candidates,
    )
    h3c_on, _ = apply_hover_specialist(h3a_prediction, condition_prediction)
    h3c_enabled, h3c_scores = select_binary_switch(
        selection_labels, h3a_prediction, h3c_on,
    )

    score_digest = _json_sha256(
        {
            "horizontal": horizontal_scores,
            "h1": h1_scores,
            "h3b": h3b_scores,
            "h3c": h3c_scores,
        }
    )
    config: dict[str, Any] = {
        "outer_h": outer_h,
        "predicted_d": predicted_d,
        "forbidden_groups": [outer_h, predicted_d],
        "selection_label_groups": list(remaining),
        "final_fit_groups": list(remaining),
        "selection_oof_fit_groups": fit_sets,
        "selection_rows": 126,
        "selected_representation": selected_representation,
        "selected_alpha": float(selected_alpha),
        "H1_switch": "on" if h1_enabled else "off",
        "H3b_threshold": threshold_label(h3b_threshold),
        "H3c_hover_switch": "on" if h3c_enabled else "off",
        "selection_score_digest": score_digest,
        "label_access_groups": list(remaining),
        "label_read_count_by_forbidden_group": {outer_h: 0, predicted_d: 0},
        "tie_break_inputs": "frozen candidate order and numeric source-R scores only",
    }
    config["config_sha256"] = _json_sha256(config)
    oof = {
        "selection_indices": selection_indices,
        "base_prediction": base_prediction,
        "strict_prediction": strict_prediction,
        "strict_probability": strict_probability,
        "condition_prediction": condition_prediction,
        "H1_prediction": h1_prediction,
        "H3a_prediction": h3a_prediction,
    }
    return config, oof


def _predict_pair_d(
    *,
    config: Mapping[str, Any],
    group: np.ndarray,
    c2_gate: np.ndarray,
    base_cache: StrictFiveSixBaseCache,
    horizontal_cache: StrictFiveSixHorizontalCache,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    predicted_d = str(config["predicted_d"])
    fit_groups = tuple(str(name) for name in config["final_fit_groups"])
    indices = np.flatnonzero(group == predicted_d)
    if len(indices) != 21:
        raise RuntimeError("predicted source domain must contain 21 records")
    evidence = base_cache.get(fit_groups, predicted_d)
    base_prediction, _, _ = router.plurality_condition_tie(evidence)
    horizontal = horizontal_cache.get(
        fit_groups,
        predicted_d,
        str(config["selected_representation"]),
        float(config["selected_alpha"]),
    )
    h1_on, _ = apply_horizontal_override(
        base_prediction, horizontal.prediction, c2_gate[indices],
    )
    h1 = h1_on if config["H1_switch"] == "on" else base_prediction.copy()
    strict = evidence[STRICT_BRANCH]
    condition = evidence[AGREEMENT_BRANCH]
    h3a, _ = apply_strict_veto(h1, strict.prediction, strict.probability)
    selected_threshold = next(
        threshold
        for threshold in VGAP_THRESHOLDS
        if threshold_label(threshold) == config["H3b_threshold"]
    )
    h3b, _ = apply_strict_veto(
        h1,
        strict.prediction,
        strict.probability,
        threshold=selected_threshold,
    )
    h3c_on, _ = apply_hover_specialist(h3a, condition.prediction)
    h3c = h3c_on if config["H3c_hover_switch"] == "on" else h3a.copy()
    output = {
        "StrictMain_prediction": strict.prediction.copy(),
        "StrictMain_probability": strict.probability.copy(),
        "ConditionPrototype_prediction": condition.prediction.copy(),
        "H1_prediction": h1,
        "H3a_prediction": h3a,
        "H3b_prediction": h3b,
        "H3c_prediction": h3c,
    }
    final_inputs = {
        "base_prediction": base_prediction,
        "horizontal_prediction": horizontal.prediction,
        "c2_gate": c2_gate[indices],
        "strict_prediction": strict.prediction,
        "strict_probability": strict.probability,
        "condition_prediction": condition.prediction,
    }
    return output, final_inputs


def _pair_output_sha256(output: Mapping[str, np.ndarray]) -> str:
    return _json_sha256(
        {name: _array_sha256(output[name]) for name in sorted(output)}
    )


def _permutation_audit(
    *,
    config: Mapping[str, Any],
    output: Mapping[str, np.ndarray],
    final_inputs: Mapping[str, np.ndarray],
    event_d: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    pair_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}|{config['outer_h']}|{config['predicted_d']}".encode("utf-8")
        ).digest()[:8],
        "little",
    )
    permutation = np.random.default_rng(pair_seed).permutation(len(event_d))
    inverse = np.argsort(permutation)
    permuted_inputs = {name: value[permutation] for name, value in final_inputs.items()}
    base = permuted_inputs["base_prediction"]
    horizontal = permuted_inputs["horizontal_prediction"]
    h1_on, _ = apply_horizontal_override(
        base, horizontal, permuted_inputs["c2_gate"],
    )
    h1 = h1_on if config["H1_switch"] == "on" else base.copy()
    h3a, _ = apply_strict_veto(
        h1,
        permuted_inputs["strict_prediction"],
        permuted_inputs["strict_probability"],
    )
    threshold = next(
        value
        for value in VGAP_THRESHOLDS
        if threshold_label(value) == config["H3b_threshold"]
    )
    h3b, _ = apply_strict_veto(
        h1,
        permuted_inputs["strict_prediction"],
        permuted_inputs["strict_probability"],
        threshold=threshold,
    )
    h3c_on, _ = apply_hover_specialist(
        h3a, permuted_inputs["condition_prediction"],
    )
    h3c = h3c_on if config["H3c_hover_switch"] == "on" else h3a.copy()
    permuted_output = {
        "StrictMain_prediction": permuted_inputs["strict_prediction"],
        "StrictMain_probability": permuted_inputs["strict_probability"],
        "ConditionPrototype_prediction": permuted_inputs["condition_prediction"],
        "H1_prediction": h1,
        "H3a_prediction": h3a,
        "H3b_prediction": h3b,
        "H3c_prediction": h3c,
    }
    equal = {
        name: bool(np.array_equal(permuted_output[name][inverse], output[name]))
        for name in output
    }
    if not all(equal.values()):
        raise RuntimeError("record permutation equivariance audit failed")
    return {
        "permutation_sha256": _array_sha256(permutation.astype(np.int64)),
        "permuted_event_sha256": _array_sha256(event_d[permutation]),
        "permuted_input_logic_sha256": _json_sha256(
            {name: _array_sha256(value) for name, value in permuted_inputs.items()}
        ),
        "inverse_aligned_equal": equal,
        "scope": (
            "synchronized final per-record evidence and fixed local gates; all "
            "models were fit without d and no d-cohort aggregate enters composition"
        ),
    }


def _generate(
    *,
    data: Mapping[str, Any],
    periodic: Mapping[str, Any],
    seed: int,
    poison_audit: bool,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    event = np.asarray(data["event"]).astype(str)
    group = np.asarray(data["group"]).astype(str)
    target = np.asarray(data["target"], dtype=np.int64)
    groups = _validate_inventory(event, group, target)
    base_cache, horizontal_cache = _make_caches(
        data, periodic["representations"], seed=seed,
    )
    meta_source_mask = np.zeros((8, len(event)), dtype=bool)
    integer_output = {
        name: np.full((8, len(event)), -1, dtype=np.int64)
        for name in (
            "meta_StrictMain_prediction",
            "meta_ConditionPrototype_prediction",
            "meta_H1_prediction",
            "meta_H3a_prediction",
            "meta_H3b_prediction",
            "meta_H3c_prediction",
        )
    }
    probability = np.full((8, len(event), 5), np.nan, dtype=np.float32)
    pair_audits: list[dict[str, Any]] = []
    selection_manifest: list[dict[str, Any]] = []
    completed = 0
    for fold_index, outer_h in enumerate(groups):
        for predicted_d in groups:
            if predicted_d == outer_h:
                continue
            if pair_limit is not None and completed >= pair_limit:
                break
            config, _ = _select_pair_config(
                outer_h=outer_h,
                predicted_d=predicted_d,
                groups=groups,
                group=group,
                selection_target=target,
                c2_gate=periodic["c2_gate"],
                representations=periodic["representations"],
                base_cache=base_cache,
                horizontal_cache=horizontal_cache,
            )
            pair_output, final_inputs = _predict_pair_d(
                config=config,
                group=group,
                c2_gate=periodic["c2_gate"],
                base_cache=base_cache,
                horizontal_cache=horizontal_cache,
            )
            local = np.flatnonzero(group == predicted_d)
            meta_source_mask[fold_index, local] = True
            probability[fold_index, local] = pair_output["StrictMain_probability"]
            for meta_name, output_name in (
                ("meta_StrictMain_prediction", "StrictMain_prediction"),
                ("meta_ConditionPrototype_prediction", "ConditionPrototype_prediction"),
                ("meta_H1_prediction", "H1_prediction"),
                ("meta_H3a_prediction", "H3a_prediction"),
                ("meta_H3b_prediction", "H3b_prediction"),
                ("meta_H3c_prediction", "H3c_prediction"),
            ):
                integer_output[meta_name][fold_index, local] = pair_output[output_name]

            baseline_output_hash = _pair_output_sha256(pair_output)
            pair_audit: dict[str, Any] = {
                **config,
                "predicted_rows": 21,
                "output_sha256": baseline_output_hash,
                "feature_direction": {
                    "configuration_function_reads_d_features": False,
                    "configuration_function_reads_h_features": False,
                    "final_forward_reads_d_features": True,
                    "h_or_d_in_any_fit_group": False,
                    "R_feature_groups_in_fit_and_selection": list(config["final_fit_groups"]),
                    "shared_forward_batches_are_row_separable": True,
                },
                "permutation_audit": _permutation_audit(
                    config=config,
                    output=pair_output,
                    final_inputs=final_inputs,
                    event_d=event[local],
                    seed=seed,
                ),
            }

            if poison_audit:
                pair_seed = int.from_bytes(
                    hashlib.sha256(
                        f"poison|{seed}|{outer_h}|{predicted_d}".encode("utf-8")
                    ).digest()[:8],
                    "little",
                )
                rng = np.random.default_rng(pair_seed)
                variants: dict[str, np.ndarray] = {}
                h_permuted = target.copy()
                h_rows = np.flatnonzero(group == outer_h)
                h_permuted[h_rows] = h_permuted[h_rows][rng.permutation(len(h_rows))]
                variants["permute_h_labels"] = h_permuted
                d_permuted = target.copy()
                d_rows = np.flatnonzero(group == predicted_d)
                d_permuted[d_rows] = d_permuted[d_rows][rng.permutation(len(d_rows))]
                variants["permute_d_labels"] = d_permuted
                sentinel = target.copy()
                sentinel[np.isin(group, (outer_h, predicted_d))] = POISON_SENTINEL
                variants["sentinel_h_and_d"] = sentinel
                poison_rows: dict[str, Any] = {}
                for name, poisoned_target in variants.items():
                    poison_config, _ = _select_pair_config(
                        outer_h=outer_h,
                        predicted_d=predicted_d,
                        groups=groups,
                        group=group,
                        selection_target=poisoned_target,
                        c2_gate=periodic["c2_gate"],
                        representations=periodic["representations"],
                        base_cache=base_cache,
                        horizontal_cache=horizontal_cache,
                    )
                    poison_output, _ = _predict_pair_d(
                        config=poison_config,
                        group=group,
                        c2_gate=periodic["c2_gate"],
                        base_cache=base_cache,
                        horizontal_cache=horizontal_cache,
                    )
                    output_hash = _pair_output_sha256(poison_output)
                    equal = bool(
                        poison_config["config_sha256"] == config["config_sha256"]
                        and output_hash == baseline_output_hash
                        and all(
                            np.array_equal(poison_output[key], pair_output[key])
                            for key in pair_output
                        )
                    )
                    if not equal:
                        raise RuntimeError(f"label poison invariance failed: {name}")
                    poison_rows[name] = {
                        "poisoned_label_logic_sha256": _array_sha256(poisoned_target),
                        "config_sha256": poison_config["config_sha256"],
                        "output_sha256": output_hash,
                        "byte_equal_to_baseline": True,
                    }
                pair_audit["label_poison_audit"] = poison_rows
            selection_manifest.append(config)
            pair_audits.append(pair_audit)
            completed += 1
            print(
                f"strict-meta pair={completed}/{pair_limit or 56} h={outer_h} d={predicted_d} "
                f"rep={config['selected_representation']} a={config['selected_alpha']:g} "
                f"H1={config['H1_switch']} H3b={config['H3b_threshold']} "
                f"H3c={config['H3c_hover_switch']}",
                flush=True,
            )
        if pair_limit is not None and completed >= pair_limit:
            break
    arrays = {
        **integer_output,
        "meta_StrictMain_probability": probability,
        "meta_source_mask": meta_source_mask,
    }
    return {
        "arrays": arrays,
        "pair_audits": pair_audits,
        "selection_manifest": selection_manifest,
        "base_fit_audit": list(base_cache.fit_audit.values()),
        "horizontal_fit_audit": list(horizontal_cache.fit_audit.values()),
        "completed_pairs": completed,
    }
