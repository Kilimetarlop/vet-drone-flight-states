"""Per-recording periodicity statistics and the gate of the horizontal expert.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np

from .slow_time_motion import RAW_WINDOW_SIZES, q90_gap_nyquist_proxy_hz, recording_local_residualize
from .slow_time_features import load_slow_time_cache


SCRIPT = Path(__file__).resolve()


WORKSPACE = SCRIPT.parents[1]


METHOD_VERSION = "drff-r2-behavior5-c2-periodic-observability-gate/v1"


REPORT_SCHEMA = "drff-r2-dataset1-behavior5-c2-periodic-observability/v1"


FREQUENCY_LOW_HZ = 10.0


FREQUENCY_HIGH_HZ = 90.0


FREQUENCY_STEP_HZ = 0.5


TIME_SHUFFLES = 64


DELETE_RESAMPLES = 32


DELETE_FRACTION = 0.10


ADVANTAGE_GATE_DB = 3.0


DRIFT_GATE_PERCENT = 10.0


SEED = 20260830


EXPECTED_RECORDINGS = 168


EPSILON = 1.0e-12


DEFAULT_INPUT = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_slow_time_energy_v3.npz"


DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs/behavior_recognition/drff_r2_dataset1_behavior5_periodic_observability_c2_v1"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _record_seed(event: str, branch: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{branch}:{event}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def batched_masked_lomb_power(
    timestamp_seconds: np.ndarray,
    signal_rows: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Generalized Lomb explained power for batched values and/or masks."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    value = np.asarray(signal_rows, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != len(timestamp):
        raise ValueError("batched Lomb values and timestamps are not aligned")
    if len(timestamp) < 3 or len(frequency) < 1 or np.any(frequency <= 0.0):
        raise ValueError("batched Lomb requires positive frequencies and three samples")
    if not (
        np.isfinite(timestamp).all()
        and np.isfinite(value).all()
        and np.isfinite(frequency).all()
    ):
        raise ValueError("batched Lomb inputs contain non-finite values")
    if masks is None:
        mask = np.ones_like(value, dtype=np.float64)
    else:
        mask = np.asarray(masks, dtype=np.float64)
        if mask.ndim == 1:
            mask = mask[None]
        if mask.shape[1] != len(timestamp):
            raise ValueError("batched Lomb mask and timestamps are not aligned")
        if value.shape[0] == 1 and mask.shape[0] > 1:
            value = np.broadcast_to(value, mask.shape)
        if mask.shape != value.shape or np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("batched Lomb masks must be aligned binary rows")
    count = np.sum(mask, axis=1)
    if np.any(count < 3):
        raise ValueError("every Lomb row requires at least three retained samples")

    phase = 2.0 * np.pi * frequency[:, None] * (timestamp - timestamp[0])[None]
    cosine = np.cos(phase)
    sine = np.sin(phase)
    sum_c = mask @ cosine.T
    sum_s = mask @ sine.T
    cc = mask @ (cosine * cosine).T - sum_c * sum_c / count[:, None]
    ss = mask @ (sine * sine).T - sum_s * sum_s / count[:, None]
    cs = mask @ (cosine * sine).T - sum_c * sum_s / count[:, None]
    weighted_value = mask * value
    sum_y = np.sum(weighted_value, axis=1)
    yc = weighted_value @ cosine.T - sum_y[:, None] * sum_c / count[:, None]
    ys = weighted_value @ sine.T - sum_y[:, None] * sum_s / count[:, None]
    total = (
        np.sum(mask * value * value, axis=1)
        - sum_y * sum_y / count
    )
    determinant = cc * ss - cs * cs
    numerator = ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys
    valid = (determinant > EPSILON) & (total[:, None] > EPSILON)
    power = np.zeros_like(numerator)
    power[valid] = numerator[valid] / determinant[valid] / np.broadcast_to(
        total[:, None], numerator.shape
    )[valid]
    return np.clip(power, 0.0, 1.0)


def record_periodic_observability(
    timestamp_seconds: np.ndarray,
    residual_energy: np.ndarray,
    *,
    event_key: str,
    shuffles: int = TIME_SHUFFLES,
    delete_resamples: int = DELETE_RESAMPLES,
) -> dict[str, Any]:
    """Compute one label-blind recording-level C2 observability statistic."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    energy = np.asarray(residual_energy, dtype=np.float64).reshape(-1)
    if timestamp.shape != energy.shape or len(timestamp) < 10:
        raise ValueError("C2 requires at least ten aligned slow-time samples")
    if np.any(np.diff(timestamp) <= 0.0) or not (
        np.isfinite(timestamp).all() and np.isfinite(energy).all()
    ):
        raise ValueError("C2 timestamps must be finite and strictly increasing")
    if shuffles < 20 or delete_resamples < 10:
        raise ValueError("C2 null/resampling counts are below the fixed audit floor")

    q90_proxy = q90_gap_nyquist_proxy_hz(timestamp)
    upper_hz = min(FREQUENCY_HIGH_HZ, q90_proxy)
    if upper_hz < FREQUENCY_LOW_HZ:
        raise RuntimeError("recording does not reach the 10 Hz C2 lower edge")
    frequency = np.arange(
        FREQUENCY_LOW_HZ,
        upper_hz + 0.5 * FREQUENCY_STEP_HZ,
        FREQUENCY_STEP_HZ,
        dtype=np.float64,
    )
    if frequency[-1] > upper_hz + 1.0e-9:
        frequency = frequency[:-1]
    real_power = batched_masked_lomb_power(timestamp, energy, frequency)[0]
    real_peak_index = int(np.argmax(real_power))
    real_peak_power = float(real_power[real_peak_index])
    real_peak_hz = float(frequency[real_peak_index])

    shuffle_rng = np.random.default_rng(_record_seed(event_key, "time_shuffle"))
    shuffled_value = np.stack(
        [energy[shuffle_rng.permutation(len(energy))] for _ in range(shuffles)]
    )
    shuffled_power = batched_masked_lomb_power(
        timestamp, shuffled_value, frequency
    )
    shuffled_peak = np.max(shuffled_power, axis=1)
    null_median = float(np.median(shuffled_peak))
    advantage_db = float(
        10.0 * np.log10((real_peak_power + EPSILON) / (null_median + EPSILON))
    )
    empirical_p = float(
        (1 + int(np.sum(shuffled_peak >= real_peak_power))) / (shuffles + 1)
    )

    delete_rng = np.random.default_rng(_record_seed(event_key, "delete_10pct"))
    delete_count = max(1, int(round(DELETE_FRACTION * len(energy))))
    masks = np.ones((delete_resamples, len(energy)), dtype=np.float64)
    for row in range(delete_resamples):
        removed = delete_rng.choice(len(energy), size=delete_count, replace=False)
        masks[row, removed] = 0.0
    delete_power = batched_masked_lomb_power(
        timestamp, energy, frequency, masks=masks
    )
    delete_peak_hz = frequency[np.argmax(delete_power, axis=1)]
    relative_drift = np.abs(delete_peak_hz - real_peak_hz) / max(
        real_peak_hz, FREQUENCY_STEP_HZ
    ) * 100.0
    return {
        "burst_samples": int(len(timestamp)),
        "q90_gap_nyquist_proxy_hz": float(q90_proxy),
        "full_10_90hz_eligible": bool(q90_proxy >= FREQUENCY_HIGH_HZ),
        "spectrum_low_hz": FREQUENCY_LOW_HZ,
        "spectrum_high_hz": float(frequency[-1]),
        "frequency_bins": int(len(frequency)),
        "real_peak_hz": real_peak_hz,
        "real_peak_power": real_peak_power,
        "shuffle_peak_power_median": null_median,
        "shuffle_peak_power_q95": float(np.quantile(shuffled_peak, 0.95)),
        "real_vs_shuffle_median_peak_advantage_db": advantage_db,
        "shuffle_empirical_p": empirical_p,
        "delete_10pct_peak_drift_median_percent": float(np.median(relative_drift)),
        "delete_10pct_peak_drift_q90_percent": float(np.quantile(relative_drift, 0.90)),
        "delete_10pct_peak_same_bin_fraction": float(
            np.mean(delete_peak_hz == real_peak_hz)
        ),
        "passes_advantage_gate": bool(advantage_db >= ADVANTAGE_GATE_DB),
        "passes_drift_gate": bool(np.median(relative_drift) <= DRIFT_GATE_PERCENT),
        "passes_both_gates": bool(
            advantage_db >= ADVANTAGE_GATE_DB
            and np.median(relative_drift) <= DRIFT_GATE_PERCENT
        ),
    }


def _distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {key: float("nan") for key in ("minimum", "q10", "median", "q90", "maximum", "mean")}
    return {
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in sorted({str(row[key]) for row in rows}):
        local = [row for row in rows if str(row[key]) == group]
        eligible = [row for row in local if row["full_10_90hz_eligible"]]
        result[group] = {
            "recordings": len(local),
            "full_10_90hz_eligible": len(eligible),
            "full_10_90hz_coverage_fraction": len(eligible) / len(local),
            "eligible_advantage_db": _distribution(
                [row["real_vs_shuffle_median_peak_advantage_db"] for row in eligible]
            ),
            "eligible_delete_drift_percent": _distribution(
                [row["delete_10pct_peak_drift_median_percent"] for row in eligible]
            ),
            "eligible_pass_both_count": int(
                sum(row["passes_both_gates"] for row in eligible)
            ),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = load_slow_time_cache(args.input)
    names = source["burst_feature_names"]
    index = {name: names.index(name) for name in names}
    required = {
        "center_offset_hz",
        "occupied_bandwidth_hz",
        "band_excess_db_n1024",
        "band_excess_db_n2048",
    }
    if not required <= set(names):
        raise RuntimeError(f"v3 cache lacks C2 inputs: {sorted(required - set(names))}")

    record_statistics: list[dict[str, Any]] = []
    for record_index, event in enumerate(source["event"]):
        first = int(source["record_offsets"][record_index])
        stop = int(source["record_offsets"][record_index + 1])
        bursts = source["burst_features"][first:stop]
        timestamp = (
            source["window_start_sample"][first:stop]
            + max(RAW_WINDOW_SIZES) // 2
        ).astype(np.float64) / source["sample_rate_hz"]
        center = bursts[:, index["center_offset_hz"]]
        bandwidth = bursts[:, index["occupied_bandwidth_hz"]]
        excess_1024 = bursts[:, index["band_excess_db_n1024"]]
        excess_2048 = bursts[:, index["band_excess_db_n2048"]]
        energy = np.column_stack(
            (excess_1024, excess_2048, 0.5 * (excess_1024 + excess_2048))
        )
        residual = recording_local_residualize(
            energy,
            center,
            bandwidth,
            sample_rate_hz=source["sample_rate_hz"],
        )
        statistic = record_periodic_observability(
            timestamp,
            residual[:, 2],
            event_key=str(event),
        )
        statistic.update(
            {
                "event": str(event),
                "identity": str(source["identity"][record_index]),
                "behavior_index": int(source["target"][record_index]),
                "behavior": source["class_names"][
                    int(source["target"][record_index])
                ],
            }
        )
        record_statistics.append(statistic)

    if len(record_statistics) != EXPECTED_RECORDINGS:
        raise RuntimeError("C2 record-level audit is incomplete")
    eligible = [row for row in record_statistics if row["full_10_90hz_eligible"]]
    if not eligible:
        raise RuntimeError("no recording has full 10--90 Hz q90-gap coverage")
    median_advantage = float(
        np.median(
            [row["real_vs_shuffle_median_peak_advantage_db"] for row in eligible]
        )
    )
    median_drift = float(
        np.median(
            [row["delete_10pct_peak_drift_median_percent"] for row in eligible]
        )
    )
    proceed = bool(
        median_advantage >= ADVANTAGE_GATE_DB
        and median_drift <= DRIFT_GATE_PERCENT
    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "record_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(record_statistics[0].keys())
        )
        writer.writeheader()
        writer.writerows(record_statistics)
    report = {
        "schema": REPORT_SCHEMA,
        "method_version": METHOD_VERSION,
        "status": "complete",
        "decision": {
            "recommend_enter_nudft_periodic_source_separation": proceed,
            "verdict": "PROCEED_C2" if proceed else "REJECT_C2",
            "rule": "eligible-cohort median advantage >= 3 dB AND median delete-10% peak drift <= 10%",
            "observed_median_advantage_db": median_advantage,
            "observed_median_delete_10pct_peak_drift_percent": median_drift,
            "passes_advantage_gate": median_advantage >= ADVANTAGE_GATE_DB,
            "passes_drift_gate": median_drift <= DRIFT_GATE_PERCENT,
        },
        "fixed_protocol": {
            "classifier_trained": False,
            "primary_frequency_band_hz": [FREQUENCY_LOW_HZ, FREQUENCY_HIGH_HZ],
            "frequency_step_hz": FREQUENCY_STEP_HZ,
            "primary_eligibility": "record q90-gap Nyquist proxy >= 90 Hz",
            "ineligible_record_handling": "compute 10 Hz to own q90-gap proxy; exclude only from primary gate median",
            "energy_channel": "record-local residualized mean of N=1024 and N=2048 band-minus-adjacent-band excess",
            "time_shuffles": TIME_SHUFFLES,
            "shuffle_null_statistic": "maximum generalized-Lomb explained power in each record's audited band",
            "advantage_db": "10*log10(real maximum peak / median shuffled maximum peak)",
            "delete_10pct_resamples": DELETE_RESAMPLES,
            "peak_drift": "median abs(resampled peak Hz - real peak Hz) / real peak Hz * 100",
            "seed": SEED,
            "advantage_gate_db": ADVANTAGE_GATE_DB,
            "drift_gate_percent": DRIFT_GATE_PERCENT,
        },
        "leakage_audit": {
            "record_statistic_inputs": [
                "exact int64-derived burst timestamps",
                "band-minus-adjacent-band energy",
                "representative center frequency",
                "representative occupied bandwidth",
            ],
            "identity_behavior_condition_used_to_compute_statistics": False,
            "identity_and_behavior_attached_after_record_statistics": True,
            "labels_used_for_group_coverage_reporting_only": True,
            "target_cohort_statistics_used": False,
            "settings_tuned_after_results": False,
        },
        "coverage_and_primary_summary": {
            "recordings": len(record_statistics),
            "full_10_90hz_eligible": len(eligible),
            "full_10_90hz_coverage_fraction": len(eligible) / len(record_statistics),
            "eligible_advantage_db": _distribution(
                [row["real_vs_shuffle_median_peak_advantage_db"] for row in eligible]
            ),
            "eligible_delete_drift_percent": _distribution(
                [row["delete_10pct_peak_drift_median_percent"] for row in eligible]
            ),
            "eligible_shuffle_empirical_p": _distribution(
                [row["shuffle_empirical_p"] for row in eligible]
            ),
            "eligible_pass_both_count": int(
                sum(row["passes_both_gates"] for row in eligible)
            ),
        },
        "by_identity": _group_summary(record_statistics, "identity"),
        "by_behavior": _group_summary(record_statistics, "behavior"),
        "record_statistics": record_statistics,
        "artifacts": {
            "source_cache": str(args.input.resolve()),
            "source_cache_sha256": source["cache_sha256"],
            "record_statistics_csv": str(csv_path.resolve()),
        },
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    return {
        "status": "complete",
        "verdict": report["decision"]["verdict"],
        "report": str(report_path.resolve()),
        "full_band_eligible": len(eligible),
        "median_advantage_db": median_advantage,
        "median_delete_drift_percent": median_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
