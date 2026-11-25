"""Builds the burst-event cache of every recording.
"""

from __future__ import annotations

from .dataset import DEFAULT_DATASET_ROOT as _DATASET_ROOT  # set DRFF_R2_DATASET1 to the Dataset 1 folder

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable
import h5py
import numpy as np
import torch

from .dataset import BEHAVIOR5_CLASSES, load_drff_r2_dataset1_behavior5_inventory
from .event_detector import RFEvent, RFEventDetection, RFEventDetectorConfig, detect_rf_events


SCRIPT = Path(__file__).resolve()


WORKSPACE = SCRIPT.parents[1]


METHOD_VERSION = "drff-r2-behavior5-event-detector-v2/v3"


CACHE_SCHEMA = "drff-r2-dataset1-behavior5-event-token-v2/v3"


REPORT_SCHEMA = "drff-r2-dataset1-behavior5-event-audit-v2/v3"


PARTIAL_SCHEMA = "drff-r2-dataset1-behavior5-event-token-v2-partial/v3"


EXPECTED_RECORDINGS = 168


EXPECTED_SAMPLES = 140_000_000


DEFAULT_DATASET_ROOT = _DATASET_ROOT


DEFAULT_PARTIAL = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_event_tokens_v2.partial"


DEFAULT_CACHE = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_event_tokens_v2.npz"


DEFAULT_REPORT = (
    WORKSPACE
    / "outputs/behavior_recognition/drff_r2_dataset1_behavior5_event_tokens_v2/report.json"
)


DEFAULT_CRC_REPORTS = (
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_c1_c7_o2_integer_cfo.json",
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_curved_c8_c12_o2_integer_cfo.json",
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_descend_c19_c21_o2_integer_cfo.json",
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_hover_c13_c15_o2_integer_cfo.json",
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_linear_c2_c6_o2_integer_cfo.json",
    WORKSPACE / "outputs/protocol/drff_r2_dataset1_air2_takeoff_c16_c18_o2_integer_cfo.json",
)


EVENT_FEATURE_NAMES = (
    "start_seconds",
    "duration_seconds",
    "prior_start_iat_seconds",
    "prior_silence_gap_seconds",
    "time_center_fraction",
    "center_offset_hz",
    "occupied_bandwidth_hz",
    "center_drift_hz",
    "active_pixel_fraction",
    "mean_cfar_db",
    "peak_cfar_db",
    "mean_noise_excess_db",
    "peak_noise_excess_db",
    "spectral_entropy",
    "spectral_flatness",
    "log_active_energy_proxy",
)


RECORD_DIAGNOSTIC_NAMES = (
    "event_count",
    "event_rate_hz",
    "active_frame_fraction",
    "active_pixel_fraction",
    "summed_event_duration_seconds",
    "duration_median_seconds",
    "duration_q90_seconds",
    "iat_median_seconds",
    "iat_q90_seconds",
    "bandwidth_median_hz",
    "bandwidth_q90_hz",
    "center_jump_median_hz",
    "mean_cfar_median_db",
    "noise_floor_median_db",
    "noise_floor_iqr_db",
    "event_count_threshold_minus_0p5db",
    "event_count_threshold_plus_0p5db",
    "active_fraction_threshold_minus_0p5db",
    "active_fraction_threshold_plus_0p5db",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


@torch.inference_mode()
def subband_power_frames(
    real: np.ndarray,
    imag: np.ndarray,
    *,
    config: RFEventDetectorConfig,
    device: torch.device,
    taper: torch.Tensor | None = None,
) -> np.ndarray:
    """Convert complete non-overlapping FFT frames to linear subband power."""

    real_value = np.asarray(real, dtype=np.float32)
    imag_value = np.asarray(imag, dtype=np.float32)
    if real_value.ndim != 1 or imag_value.shape != real_value.shape:
        raise ValueError("I/Q arrays must be aligned and one-dimensional")
    if len(real_value) == 0 or len(real_value) % config.fft_size:
        raise ValueError("I/Q chunk must contain complete FFT frames")
    frame_count = len(real_value) // config.fft_size
    signal = torch.complex(
        torch.as_tensor(real_value, device=device),
        torch.as_tensor(imag_value, device=device),
    ).reshape(frame_count, config.fft_size)
    signal = signal - signal.mean(dim=1, keepdim=True)
    window = (
        torch.hann_window(config.fft_size, periodic=True, device=device)
        if taper is None
        else taper
    )
    if window.shape != (config.fft_size,) or window.device != signal.device:
        raise ValueError("taper is incompatible with the detector configuration")
    spectrum = torch.fft.fftshift(
        torch.fft.fft(signal * window, dim=1), dim=1
    )
    bins_per_subband = config.fft_size // config.subbands
    power = spectrum.abs().square().reshape(
        frame_count, config.subbands, bins_per_subband
    ).mean(dim=2)
    result = power.float().cpu().numpy()
    if result.shape != (frame_count, config.subbands):
        raise RuntimeError("subband power shape drift")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise RuntimeError("subband power contains invalid values")
    return result


def extract_record_power(
    path: Path,
    *,
    config: RFEventDetectorConfig,
    device: torch.device,
    frames_per_read: int,
) -> np.ndarray:
    """Read one recording sequentially and retain only its subband power map."""

    frame_count = EXPECTED_SAMPLES // config.fft_size
    result = np.empty((frame_count, config.subbands), dtype=np.float32)
    taper = torch.hann_window(config.fft_size, periodic=True, device=device)
    with h5py.File(path, "r") as source:
        i_data, q_data = source["RF0_I"], source["RF0_Q"]
        if i_data.shape != (EXPECTED_SAMPLES, 1) or q_data.shape != i_data.shape:
            raise RuntimeError(f"unexpected I/Q shape in {path.name}")
        for first_frame in range(0, frame_count, frames_per_read):
            count = min(frames_per_read, frame_count - first_frame)
            first_sample = first_frame * config.fft_size
            stop_sample = first_sample + count * config.fft_size
            real = np.asarray(i_data[first_sample:stop_sample, 0], dtype=np.float32)
            imag = np.asarray(q_data[first_sample:stop_sample, 0], dtype=np.float32)
            result[first_frame : first_frame + count] = subband_power_frames(
                real,
                imag,
                config=config,
                device=device,
                taper=taper,
            )
    return result


def event_feature_matrix(
    events: Iterable[RFEvent],
    *,
    config: RFEventDetectorConfig,
    recording_frames: int,
) -> np.ndarray:
    rows = tuple(events)
    result = np.empty((len(rows), len(EVENT_FEATURE_NAMES)), dtype=np.float32)
    previous_start = 0
    running_max_stop = 0
    for index, event in enumerate(rows):
        start_seconds = event.start_frame * config.frame_duration_seconds
        duration_seconds = event.duration_frames * config.frame_duration_seconds
        iat_seconds = (
            0.0
            if index == 0
            else (event.start_frame - previous_start) * config.frame_duration_seconds
        )
        gap_seconds = (
            start_seconds
            if index == 0
            else max(0, event.start_frame - running_max_stop)
            * config.frame_duration_seconds
        )
        # Only detector seed pixels carry measured RF evidence.  Bounding-box
        # holes and morphological closing bridges are grouping aids and must
        # not contribute energy.
        energy_proxy = event.active_pixels * 10.0 ** (
            event.mean_noise_excess_db / 10.0
        )
        result[index] = (
            start_seconds,
            duration_seconds,
            iat_seconds,
            gap_seconds,
            0.5 * (event.start_frame + event.stop_frame) / recording_frames,
            event.center_offset_hz,
            event.occupied_bandwidth_hz,
            event.center_drift_hz,
            event.active_pixels / max(event.box_pixels, 1),
            event.mean_cfar_db,
            event.peak_cfar_db,
            event.mean_noise_excess_db,
            event.peak_noise_excess_db,
            event.spectral_entropy,
            event.spectral_flatness,
            math.log1p(energy_proxy),
        )
        previous_start = event.start_frame
        running_max_stop = max(running_max_stop, event.stop_frame)
    if not np.isfinite(result).all():
        raise RuntimeError("event feature matrix contains non-finite values")
    return result


def _quantile_or_zero(value: np.ndarray, quantile: float) -> float:
    return float(np.quantile(value, quantile)) if len(value) else 0.0


def record_diagnostics(
    features: np.ndarray,
    detection: RFEventDetection,
    lower: RFEventDetection,
    upper: RFEventDetection,
    *,
    recording_seconds: float,
) -> np.ndarray:
    index = {name: EVENT_FEATURE_NAMES.index(name) for name in EVENT_FEATURE_NAMES}
    duration = features[:, index["duration_seconds"]]
    iat = features[1:, index["prior_start_iat_seconds"]]
    bandwidth = features[:, index["occupied_bandwidth_hz"]]
    center = features[:, index["center_offset_hz"]]
    center_jump = np.abs(np.diff(center))
    mean_cfar = features[:, index["mean_cfar_db"]]
    noise_q25, noise_q75 = np.quantile(detection.noise_floor_db, (0.25, 0.75))
    values = np.asarray(
        (
            len(features),
            len(features) / recording_seconds,
            detection.active_frame_fraction,
            detection.active_pixel_fraction,
            float(duration.sum()),
            _quantile_or_zero(duration, 0.50),
            _quantile_or_zero(duration, 0.90),
            _quantile_or_zero(iat, 0.50),
            _quantile_or_zero(iat, 0.90),
            _quantile_or_zero(bandwidth, 0.50),
            _quantile_or_zero(bandwidth, 0.90),
            _quantile_or_zero(center_jump, 0.50),
            _quantile_or_zero(mean_cfar, 0.50),
            float(np.median(detection.noise_floor_db)),
            float(noise_q75 - noise_q25),
            len(lower.events),
            len(upper.events),
            lower.active_frame_fraction,
            upper.active_frame_fraction,
        ),
        dtype=np.float32,
    )
    if values.shape != (len(RECORD_DIAGNOSTIC_NAMES),):
        raise RuntimeError("record diagnostic width drift")
    return values


def _partial_path(directory: Path, index: int) -> Path:
    return directory / f"record_{index:03d}.npz"


def _partial_fingerprint(row: Any, config: RFEventDetectorConfig) -> str:
    source = Path(row.path)
    stat = source.stat()
    payload = {
        "partial_schema": PARTIAL_SCHEMA,
        "method_version": METHOD_VERSION,
        "event": row.event,
        "source": {
            "path": str(source.resolve()),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "detector_config": config.to_dict(),
        "event_feature_names": list(EVENT_FEATURE_NAMES),
        "record_diagnostic_names": list(RECORD_DIAGNOSTIC_NAMES),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_partial(
    path: Path,
    *,
    event: str,
    features: np.ndarray,
    diagnostics: np.ndarray,
    elapsed_seconds: float,
    fingerprint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            event=np.asarray(event),
            event_features=features,
            diagnostics=diagnostics,
            elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
            partial_schema=np.asarray(PARTIAL_SCHEMA),
            provenance_fingerprint=np.asarray(fingerprint),
        )
    os.replace(temporary, path)


def _load_partial(
    path: Path,
    expected_event: str,
    expected_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["event"].item()) != expected_event:
            raise RuntimeError(f"partial event mismatch: {path}")
        if str(cache["partial_schema"].item()) != PARTIAL_SCHEMA:
            raise RuntimeError(f"partial schema mismatch: {path}")
        if str(cache["provenance_fingerprint"].item()) != expected_fingerprint:
            raise RuntimeError(
                f"partial provenance mismatch; use a new partial directory: {path}"
            )
        features = np.asarray(cache["event_features"], dtype=np.float32)
        diagnostics = np.asarray(cache["diagnostics"], dtype=np.float32)
        elapsed = float(cache["elapsed_seconds"].item())
    if features.ndim != 2 or features.shape[1] != len(EVENT_FEATURE_NAMES):
        raise RuntimeError(f"partial event feature shape drift: {path}")
    if diagnostics.shape != (len(RECORD_DIAGNOSTIC_NAMES),):
        raise RuntimeError(f"partial diagnostic shape drift: {path}")
    return features, diagnostics, elapsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adjusted_eta_squared(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups).astype(str)
    grand = matrix.mean(axis=0)
    total = np.sum((matrix - grand) ** 2, axis=0)
    between = np.zeros(matrix.shape[1], dtype=np.float64)
    for group in np.unique(labels):
        selected = matrix[labels == group]
        between += len(selected) * (selected.mean(axis=0) - grand) ** 2
    eta = np.divide(between, total, out=np.zeros_like(total), where=total > 0.0)
    group_count = len(np.unique(labels))
    sample_count = len(matrix)
    adjusted = 1.0 - (1.0 - eta) * (sample_count - 1) / max(
        sample_count - group_count, 1
    )
    return np.clip(adjusted, 0.0, 1.0)


def _distance_audit(
    diagnostics: np.ndarray,
    identity: np.ndarray,
    condition: np.ndarray,
    class_index: np.ndarray,
) -> dict[str, Any]:
    selected_names = RECORD_DIAGNOSTIC_NAMES[:13]
    value = np.asarray(diagnostics[:, : len(selected_names)], dtype=np.float64)
    center = np.median(value, axis=0)
    scale = 1.4826 * np.median(np.abs(value - center), axis=0)
    keep = scale > 1.0e-9
    standardized = (value[:, keep] - center[keep]) / scale[keep]
    same_condition_cross_identity = []
    same_identity_different_behavior = []
    for left in range(len(value)):
        for right in range(left + 1, len(value)):
            distance = float(np.linalg.norm(standardized[left] - standardized[right]))
            if condition[left] == condition[right] and identity[left] != identity[right]:
                same_condition_cross_identity.append(distance)
            if identity[left] == identity[right] and class_index[left] != class_index[right]:
                same_identity_different_behavior.append(distance)
    cross = np.asarray(same_condition_cross_identity, dtype=np.float64)
    within = np.asarray(same_identity_different_behavior, dtype=np.float64)
    return {
        "selected_features": list(selected_names),
        "nonconstant_dimensions": int(keep.sum()),
        "same_condition_cross_identity_pairs": int(len(cross)),
        "same_identity_different_behavior_pairs": int(len(within)),
        "same_condition_cross_identity_median_distance": float(np.median(cross)),
        "same_identity_different_behavior_median_distance": float(np.median(within)),
        "desired_ordering_observed": bool(np.median(cross) < np.median(within)),
        "distance_ratio_cross_identity_to_within_identity": float(
            np.median(cross) / max(np.median(within), np.finfo(float).tiny)
        ),
    }


def _known_crc_windows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"required strict-CRC audit report is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for recording in payload.get("recordings", []):
            for candidate in recording.get("decode_candidates", []):
                if candidate.get("status") == "crc_valid":
                    rows.append(
                        {
                            "event": str(recording["event"]),
                            "start_sample": int(candidate["start_sample"]),
                            "end_sample": int(candidate["end_sample"]),
                            "center_offset_hz": float(candidate["center_offset_hz"]),
                            "source_report": str(path.resolve()),
                        }
                    )
    unique = {
        (row["event"], row["start_sample"], row["end_sample"]): row for row in rows
    }
    return list(unique.values())


def _crc_overlap_audit(
    crc_rows: list[dict[str, Any]],
    events: np.ndarray,
    offsets: np.ndarray,
    event_names: np.ndarray,
    *,
    config: RFEventDetectorConfig,
) -> dict[str, Any]:
    name_to_index = {str(name): index for index, name in enumerate(event_names)}
    feature_index = {name: EVENT_FEATURE_NAMES.index(name) for name in EVENT_FEATURE_NAMES}
    rows = []
    for crc in crc_rows:
        record_index = name_to_index.get(crc["event"])
        time_detected = False
        time_frequency_detected = False
        strict_anchor_matched = False
        maximum_overlap = 0.0
        maximum_temporal_iou = 0.0
        matched_center_error_hz: float | None = None
        best_iou_duration_ratio: float | None = None
        best_iou_start_error_samples: int | None = None
        best_iou_stop_error_samples: int | None = None
        if record_index is not None:
            record_events = events[offsets[record_index] : offsets[record_index + 1]]
            for feature in record_events:
                start = int(round(feature[feature_index["start_seconds"]] * config.sample_rate_hz))
                stop = int(
                    round(
                        (feature[feature_index["start_seconds"]]
                        + feature[feature_index["duration_seconds"]])
                        * config.sample_rate_hz
                    )
                )
                overlap = max(
                    0,
                    min(stop, crc["end_sample"]) - max(start, crc["start_sample"]),
                )
                fraction = overlap / max(1, crc["end_sample"] - crc["start_sample"])
                maximum_overlap = max(maximum_overlap, fraction)
                union = max(stop, crc["end_sample"]) - min(start, crc["start_sample"])
                temporal_iou = overlap / max(1, union)
                if temporal_iou > maximum_temporal_iou:
                    maximum_temporal_iou = temporal_iou
                    best_iou_duration_ratio = (stop - start) / max(
                        1, crc["end_sample"] - crc["start_sample"]
                    )
                    best_iou_start_error_samples = start - crc["start_sample"]
                    best_iou_stop_error_samples = stop - crc["end_sample"]
                if fraction >= 0.25:
                    time_detected = True
                    center_error = abs(
                        float(feature[feature_index["center_offset_hz"]])
                        - float(crc["center_offset_hz"])
                    )
                    tolerance = (
                        0.5
                        * float(feature[feature_index["occupied_bandwidth_hz"]])
                        + config.subband_width_hz
                    )
                    matched_center_error_hz = (
                        center_error
                        if matched_center_error_hz is None
                        else min(matched_center_error_hz, center_error)
                    )
                    if center_error <= tolerance:
                        time_frequency_detected = True
                    if (
                        temporal_iou >= 0.90
                        and center_error <= config.subband_width_hz
                    ):
                        strict_anchor_matched = True
        rows.append(
            {
                **crc,
                "time_detected": time_detected,
                "time_frequency_detected": time_frequency_detected,
                "strict_anchor_matched": strict_anchor_matched,
                "maximum_overlap_fraction": maximum_overlap,
                "maximum_temporal_iou": maximum_temporal_iou,
                "matched_center_error_hz": matched_center_error_hz,
                "best_iou_event_to_crc_duration_ratio": best_iou_duration_ratio,
                "best_iou_start_error_samples": best_iou_start_error_samples,
                "best_iou_stop_error_samples": best_iou_stop_error_samples,
            }
        )
    center_errors = np.asarray(
        [
            row["matched_center_error_hz"]
            for row in rows
            if row["matched_center_error_hz"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "known_crc_windows": len(rows),
        "time_overlapped_at_least_25_percent": int(
            sum(row["time_detected"] for row in rows)
        ),
        "time_frequency_matched": int(
            sum(row["time_frequency_detected"] for row in rows)
        ),
        "strict_iou_and_center_matched": int(
            sum(row["strict_anchor_matched"] for row in rows)
        ),
        "time_recall": (
            float(np.mean([row["time_detected"] for row in rows])) if rows else None
        ),
        "time_frequency_recall": (
            float(np.mean([row["time_frequency_detected"] for row in rows]))
            if rows
            else None
        ),
        "strict_anchor_recall": (
            float(np.mean([row["strict_anchor_matched"] for row in rows]))
            if rows
            else None
        ),
        "center_error_within_one_subband": int(
            np.sum(center_errors <= config.subband_width_hz)
        ),
        "matched_center_error_hz": (
            _distribution(center_errors) if len(center_errors) else None
        ),
        "rows": rows,
    }


def _distribution(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _combine_and_report(
    args: argparse.Namespace,
    rows: tuple[Any, ...],
    config: RFEventDetectorConfig,
) -> dict[str, Any]:
    feature_parts = []
    diagnostic_parts = []
    elapsed = []
    offsets = [0]
    for index, row in enumerate(rows):
        features, diagnostics, seconds = _load_partial(
            _partial_path(args.partial_dir, index),
            row.event,
            _partial_fingerprint(row, config),
        )
        feature_parts.append(features)
        diagnostic_parts.append(diagnostics)
        elapsed.append(seconds)
        offsets.append(offsets[-1] + len(features))
    event_features = np.concatenate(feature_parts, axis=0)
    diagnostics = np.stack(diagnostic_parts).astype(np.float32)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    metadata = {
        "event": np.asarray([row.event for row in rows]),
        "identity": np.asarray([row.identity for row in rows]),
        "behavior": np.asarray([row.behavior for row in rows]),
        "class_index": np.asarray([row.class_index for row in rows], dtype=np.int64),
        "condition": np.asarray([row.condition for row in rows], dtype=np.int64),
        "class_names": np.asarray(BEHAVIOR5_CLASSES),
        "event_feature_names": np.asarray(EVENT_FEATURE_NAMES),
        "record_diagnostic_names": np.asarray(RECORD_DIAGNOSTIC_NAMES),
        "record_offsets": offsets_array,
        "event_features": event_features,
        "record_diagnostics": diagnostics,
        "sample_rate_hz": np.asarray(config.sample_rate_hz, dtype=np.float64),
        "fft_size": np.asarray(config.fft_size, dtype=np.int64),
        "hop_size": np.asarray(config.hop_size, dtype=np.int64),
        "subbands": np.asarray(config.subbands, dtype=np.int64),
        "recording_samples": np.asarray(EXPECTED_SAMPLES, dtype=np.int64),
        "schema": np.asarray(CACHE_SCHEMA),
        "method_version": np.asarray(METHOD_VERSION),
        "detector_config_json": np.asarray(
            json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **metadata)
    os.replace(temporary, args.output)

    identity = metadata["identity"]
    condition = metadata["condition"]
    class_index = metadata["class_index"]
    physical_width = RECORD_DIAGNOSTIC_NAMES.index("noise_floor_median_db")
    identity_eta = _adjusted_eta_squared(diagnostics[:, :physical_width], identity)
    behavior_eta = _adjusted_eta_squared(
        diagnostics[:, :physical_width], class_index
    )
    names = np.asarray(RECORD_DIAGNOSTIC_NAMES[:physical_width])
    count_index = RECORD_DIAGNOSTIC_NAMES.index("event_count")
    active_index = RECORD_DIAGNOSTIC_NAMES.index("active_frame_fraction")
    lower_count = RECORD_DIAGNOSTIC_NAMES.index("event_count_threshold_minus_0p5db")
    upper_count = RECORD_DIAGNOSTIC_NAMES.index("event_count_threshold_plus_0p5db")
    default_count = diagnostics[:, count_index]
    nonzero = np.maximum(default_count, 1.0)
    report = {
        "schema": REPORT_SCHEMA,
        "method_version": METHOD_VERSION,
        "status": "complete",
        "protocol": {
            "task": "five behavior recognition from raw IQ",
            "recordings": len(rows),
            "whole_recording_isolation_required_downstream": True,
            "strict_leave_one_identity_model_out_required_downstream": True,
            "cross_channel_claim": False,
            "detector_label_inputs": [],
            "event_cap": None,
            "truncated_recordings": 0,
        },
        "detector": config.to_dict(),
        "cache": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "bytes": int(args.output.stat().st_size),
            "schema": CACHE_SCHEMA,
            "event_feature_names": list(EVENT_FEATURE_NAMES),
            "record_diagnostic_names": list(RECORD_DIAGNOSTIC_NAMES),
        },
        "event_audit": {
            "total_events": int(offsets_array[-1]),
            "event_count": _distribution(default_count),
            "active_frame_fraction": _distribution(diagnostics[:, active_index]),
            "threshold_perturbation_minus_0p5db_relative_count_change": _distribution(
                np.abs(diagnostics[:, lower_count] - default_count) / nonzero
            ),
            "threshold_perturbation_plus_0p5db_relative_count_change": _distribution(
                np.abs(diagnostics[:, upper_count] - default_count) / nonzero
            ),
            "seconds_per_recording": _distribution(np.asarray(elapsed)),
        },
        "observability_audit": {
            "identity_adjusted_eta_squared_by_physical_feature": {
                str(name): float(value) for name, value in zip(names, identity_eta, strict=True)
            },
            "behavior_adjusted_eta_squared_by_physical_feature": {
                str(name): float(value) for name, value in zip(names, behavior_eta, strict=True)
            },
            "identity_adjusted_eta_squared_median": float(np.median(identity_eta)),
            "behavior_adjusted_eta_squared_median": float(np.median(behavior_eta)),
            "identity_to_behavior_adjusted_eta_squared_ratio": float(
                np.median(identity_eta)
                / max(np.median(behavior_eta), np.finfo(float).tiny)
            ),
            "distance_test": _distance_audit(
                diagnostics, identity, condition, class_index
            ),
        },
        "known_protocol_window_audit": _crc_overlap_audit(
            _known_crc_windows(args.crc_report),
            event_features,
            offsets_array,
            metadata["event"],
            config=config,
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_name(f"{args.report.name}.{os.getpid()}.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_report, args.report)
    commit_path = args.output.with_suffix(args.output.suffix + ".complete.json")
    commit = {
        "status": "complete",
        "schema": CACHE_SCHEMA,
        "method_version": METHOD_VERSION,
        "cache": str(args.output.resolve()),
        "cache_sha256": _sha256(args.output),
        "report": str(args.report.resolve()),
        "report_sha256": _sha256(args.report),
        "detector_config": config.to_dict(),
    }
    temporary_commit = commit_path.with_name(f"{commit_path.name}.{os.getpid()}.tmp")
    temporary_commit.write_text(
        json.dumps(commit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_commit, commit_path)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows = load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    if len(rows) != EXPECTED_RECORDINGS:
        raise RuntimeError("strict inventory must contain 168 behavior recordings")
    config = RFEventDetectorConfig(
        cfar_threshold_db=args.cfar_threshold_db,
        minimum_noise_excess_db=args.minimum_noise_excess_db,
    )
    config.validate()
    if config.hop_size != config.fft_size:
        raise ValueError(
            "this non-overlapping raw-IQ builder requires hop_size == fft_size"
        )
    if args.frames_per_read < 1:
        raise ValueError("frames-per-read must be positive")
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    completed_before = sum(
        _partial_path(args.partial_dir, index).exists() for index in range(len(rows))
    )
    processed_now = 0
    for index, row in enumerate(rows):
        if args.only_identity and row.identity != args.only_identity:
            continue
        if args.only_condition and row.condition != args.only_condition:
            continue
        partial = _partial_path(args.partial_dir, index)
        fingerprint = _partial_fingerprint(row, config)
        if partial.exists():
            _load_partial(partial, row.event, fingerprint)
            continue
        record_started = time.perf_counter()
        power = extract_record_power(
            Path(row.path),
            config=config,
            device=device,
            frames_per_read=args.frames_per_read,
        )
        detection = detect_rf_events(power, config)
        lower = detect_rf_events(
            power, replace(config, cfar_threshold_db=config.cfar_threshold_db - 0.5)
        )
        upper = detect_rf_events(
            power, replace(config, cfar_threshold_db=config.cfar_threshold_db + 0.5)
        )
        features = event_feature_matrix(
            detection.events,
            config=config,
            recording_frames=len(power),
        )
        diagnostics = record_diagnostics(
            features,
            detection,
            lower,
            upper,
            recording_seconds=EXPECTED_SAMPLES / config.sample_rate_hz,
        )
        seconds = time.perf_counter() - record_started
        _write_partial(
            partial,
            event=row.event,
            features=features,
            diagnostics=diagnostics,
            elapsed_seconds=seconds,
            fingerprint=fingerprint,
        )
        processed_now += 1
        print(
            f"detector-v2 {index + 1}/{len(rows)} {row.event} "
            f"events={len(features)} active={detection.active_frame_fraction:.4f} "
            f"seconds={seconds:.2f}",
            flush=True,
        )
        if args.max_recordings and processed_now >= args.max_recordings:
            break

    completed = sum(
        _partial_path(args.partial_dir, index).exists() for index in range(len(rows))
    )
    if completed != len(rows):
        return {
            "status": "partial",
            "completed_recordings": completed,
            "remaining_recordings": len(rows) - completed,
            "processed_now": processed_now,
            "partial_dir": str(args.partial_dir.resolve()),
            "device": str(device),
            "seconds": time.perf_counter() - started,
        }
    report = _combine_and_report(args, rows, config)
    return {
        "status": "complete",
        "cache": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "total_events": report["event_audit"]["total_events"],
        "completed_before": completed_before,
        "processed_now": processed_now,
        "device": str(device),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames-per-read", type=int, default=2_048)
    parser.add_argument("--cfar-threshold-db", type=float, default=6.0)
    parser.add_argument("--minimum-noise-excess-db", type=float, default=6.0)
    parser.add_argument("--max-recordings", type=int, default=0)
    parser.add_argument("--only-identity", default="")
    parser.add_argument("--only-condition", type=int, default=0)
    parser.add_argument(
        "--crc-report",
        action="append",
        type=Path,
        default=None,
        help="Known strict-CRC report used only after detection for recall auditing.",
    )
    args = parser.parse_args()
    if args.crc_report is None:
        args.crc_report = list(DEFAULT_CRC_REPORTS)
    if args.only_identity and args.only_identity not in {
        row.identity
        for row in load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    }:
        parser.error(f"unknown --only-identity: {args.only_identity}")
    if args.only_condition and not 1 <= args.only_condition <= 21:
        parser.error("--only-condition must lie in [1,21]")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
