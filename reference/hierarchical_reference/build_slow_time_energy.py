"""Builds the slow-time energy cache around the detected bursts of every recording.
"""

from __future__ import annotations

from .dataset import DEFAULT_DATASET_ROOT as _DATASET_ROOT  # set DRFF_R2_DATASET1 to the Dataset 1 folder

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import h5py
import numpy as np

from .dataset import BEHAVIOR5_CLASSES, DATASET1_IDENTITIES, load_drff_r2_dataset1_behavior5_inventory
from .slow_time_motion import RAW_WINDOW_SIZES, band_adjacent_energy, centered_window_bounds, merge_overlapping_tf_components


SCRIPT = Path(__file__).resolve()


WORKSPACE = SCRIPT.parents[1]


METHOD_VERSION = "drff-r2-behavior5-cross-burst-slow-time-energy/v3"


INPUT_SCHEMA = "drff-r2-dataset1-behavior5-event-token-v2/v3"


CACHE_SCHEMA = "drff-r2-dataset1-behavior5-slow-time-energy/v3"


PARTIAL_SCHEMA = "drff-r2-dataset1-behavior5-slow-time-energy-partial/v3"


REPORT_SCHEMA = "drff-r2-dataset1-behavior5-slow-time-energy-audit/v3"


EXPECTED_RECORDINGS = 168


EXPECTED_RECORDING_SAMPLES = 140_000_000


EXPECTED_SAMPLE_RATE_HZ = 100_000_000.0


EXPECTED_ROWS_PER_UNIT = 21


DEFAULT_DATASET_ROOT = _DATASET_ROOT


DEFAULT_EVENT_CACHE = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_event_tokens_v2.npz"


DEFAULT_PARTIAL = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_slow_time_energy_v3.partial"


DEFAULT_OUTPUT = WORKSPACE / "tmp/drff_r2_dataset1_behavior5_slow_time_energy_v3.npz"


DEFAULT_REPORT = (
    WORKSPACE
    / "outputs/behavior_recognition/drff_r2_dataset1_behavior5_slow_time_energy_v3/report.json"
)


BURST_FEATURE_NAMES = (
    "timestamp_seconds",
    "duration_seconds",
    "center_offset_hz",
    "occupied_bandwidth_hz",
    "component_count",
    "representative_mean_cfar_db",
    "signal_db_n1024",
    "adjacent_db_n1024",
    "band_excess_db_n1024",
    "signal_db_n2048",
    "adjacent_db_n2048",
    "band_excess_db_n2048",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_event_cache(path: Path) -> dict[str, Any]:
    marker = path.with_suffix(path.suffix + ".complete.json")
    if not marker.is_file():
        raise RuntimeError(f"Detector-v2 complete marker is missing: {marker}")
    commit = json.loads(marker.read_text(encoding="utf-8"))
    observed_hash = _sha256(path)
    if (
        commit.get("status") != "complete"
        or commit.get("schema") != INPUT_SCHEMA
        or commit.get("cache_sha256") != observed_hash
    ):
        raise RuntimeError("Detector-v2 complete marker or hash is invalid")
    required = {
        "event",
        "identity",
        "class_index",
        "condition",
        "class_names",
        "event_feature_names",
        "record_offsets",
        "event_features",
        "sample_rate_hz",
        "hop_size",
        "recording_samples",
        "schema",
    }
    with np.load(path, allow_pickle=False) as cache:
        missing = required - set(cache.files)
        if missing:
            raise RuntimeError(f"Detector-v2 cache lacks keys: {sorted(missing)}")
        if str(np.asarray(cache["schema"]).item()) != INPUT_SCHEMA:
            raise RuntimeError("Detector-v2 cache schema drift")
        result = {
            "event": np.asarray(cache["event"]).astype(str),
            "identity": np.asarray(cache["identity"]).astype(str),
            "class_index": np.asarray(cache["class_index"], dtype=np.int64),
            "condition": np.asarray(cache["condition"], dtype=np.int64),
            "class_names": tuple(np.asarray(cache["class_names"]).astype(str).tolist()),
            "event_feature_names": tuple(
                np.asarray(cache["event_feature_names"]).astype(str).tolist()
            ),
            "record_offsets": np.asarray(cache["record_offsets"], dtype=np.int64),
            "event_features": np.asarray(cache["event_features"], dtype=np.float32),
            "sample_rate_hz": float(np.asarray(cache["sample_rate_hz"]).item()),
            "hop_size": int(np.asarray(cache["hop_size"]).item()),
            "recording_samples": int(np.asarray(cache["recording_samples"]).item()),
            "cache_sha256": observed_hash,
            "marker": str(marker.resolve()),
        }
    if result["event"].shape != (EXPECTED_RECORDINGS,):
        raise RuntimeError("Detector-v2 recording count drift")
    if result["class_names"] != BEHAVIOR5_CLASSES:
        raise RuntimeError("Detector-v2 class order drift")
    if result["record_offsets"].shape != (EXPECTED_RECORDINGS + 1,):
        raise RuntimeError("Detector-v2 CSR offsets drift")
    if result["record_offsets"][-1] != len(result["event_features"]):
        raise RuntimeError("Detector-v2 CSR payload is incomplete")
    if result["recording_samples"] != EXPECTED_RECORDING_SAMPLES:
        raise RuntimeError("recording length drift")
    if not np.isclose(result["sample_rate_hz"], EXPECTED_SAMPLE_RATE_HZ):
        raise RuntimeError("sample-rate drift")
    groups, counts = np.unique(result["identity"], return_counts=True)
    if tuple(sorted(groups.tolist())) != tuple(sorted(DATASET1_IDENTITIES)):
        raise RuntimeError("eight-UAV inventory drift")
    if not np.all(counts == EXPECTED_ROWS_PER_UNIT):
        raise RuntimeError("each UAV must contribute 21 recordings")
    return result


def extract_record_slow_time(
    i_data: h5py.Dataset,
    q_data: h5py.Dataset,
    events: np.ndarray,
    event_feature_names: tuple[str, ...],
    *,
    sample_rate_hz: float,
    hop_size: int,
    recording_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if i_data.shape != (recording_samples, 1) or q_data.shape != i_data.shape:
        raise RuntimeError("raw-IQ dataset shape drift")
    bursts = merge_overlapping_tf_components(
        events,
        event_feature_names,
        time_quantum_seconds=hop_size / sample_rate_hz,
    )
    if len(bursts) < 3:
        raise RuntimeError("fewer than three physical bursts are observable")
    index = {name: event_feature_names.index(name) for name in event_feature_names}
    rows = np.empty((len(bursts), len(BURST_FEATURE_NAMES)), dtype=np.float32)
    window_start = np.empty(len(bursts), dtype=np.int64)
    signal_counts: list[int] = []
    adjacent_counts: list[int] = []
    for burst_index, burst in enumerate(bursts):
        # Recover integer detector-frame boundaries, then form an exact sample
        # midpoint.  No float timestamp is used for the raw-IQ seek itself.
        first_frame = int(round(burst.start_seconds * sample_rate_hz / hop_size))
        stop_frame = int(round(burst.stop_seconds * sample_rate_hz / hop_size))
        burst_start_sample = first_frame * hop_size
        burst_stop_sample = min(recording_samples, stop_frame * hop_size)
        representative = events[burst.representative_index]
        representative_first_frame = int(
            round(
                float(representative[index["start_seconds"]])
                * sample_rate_hz
                / hop_size
            )
        )
        representative_duration_frames = max(
            1,
            int(
                round(
                    float(representative[index["duration_seconds"]])
                    * sample_rate_hz
                    / hop_size
                )
            ),
        )
        # The energy window is centered inside the selected frequency
        # component, not merely at the union midpoint.  An overlap chain can
        # be longer than its representative component; sampling the union
        # midpoint would then measure the right band at the wrong time.
        center_sample = (
            representative_first_frame * hop_size
            + representative_duration_frames * hop_size // 2
        )
        timestamp_seconds = center_sample / sample_rate_hz
        first, stop = centered_window_bounds(
            timestamp_seconds,
            window_samples=max(RAW_WINDOW_SIZES),
            sample_rate_hz=sample_rate_hz,
            recording_samples=recording_samples,
        )
        if first + max(RAW_WINDOW_SIZES) // 2 != center_sample:
            raise RuntimeError(
                "representative timestamp is too close to a record edge for an exact centered window"
            )
        real = np.asarray(i_data[first:stop, 0], dtype=np.float32)
        imag = np.asarray(q_data[first:stop, 0], dtype=np.float32)
        if real.shape != (max(RAW_WINDOW_SIZES),) or imag.shape != real.shape:
            raise RuntimeError("raw-IQ slow-time window read is incomplete")
        raw = real.astype(np.complex64, copy=False)
        raw.imag = imag
        center_hz = float(representative[index["center_offset_hz"]])
        bandwidth_hz = float(representative[index["occupied_bandwidth_hz"]])
        resolution: dict[int, tuple[float, float, float, int, int]] = {}
        for count in RAW_WINDOW_SIZES:
            offset = (len(raw) - count) // 2
            resolution[count] = band_adjacent_energy(
                raw[offset : offset + count],
                sample_rate_hz=sample_rate_hz,
                center_offset_hz=center_hz,
                occupied_bandwidth_hz=bandwidth_hz,
            )
            signal_counts.append(resolution[count][3])
            adjacent_counts.append(resolution[count][4])
        rows[burst_index] = (
            timestamp_seconds,
            (burst_stop_sample - burst_start_sample) / sample_rate_hz,
            center_hz,
            bandwidth_hz,
            burst.component_count,
            float(representative[index["mean_cfar_db"]]),
            resolution[1024][0],
            resolution[1024][1],
            resolution[1024][2],
            resolution[2048][0],
            resolution[2048][1],
            resolution[2048][2],
        )
        window_start[burst_index] = first
    timestamp = rows[:, 0]
    if np.any(np.diff(timestamp) <= 0.0):
        raise RuntimeError("merged physical-burst timestamps are not strictly increasing")
    if not np.isfinite(rows).all():
        raise RuntimeError("slow-time burst cache contains non-finite values")
    return rows, window_start, {
        "detector_tf_components": int(len(events)),
        "physical_time_bursts": int(len(bursts)),
        "raw_complex_samples_read": int(len(bursts) * max(RAW_WINDOW_SIZES)),
        "first_timestamp_seconds": float(timestamp[0]),
        "last_timestamp_seconds": float(timestamp[-1]),
        "minimum_signal_bins": int(min(signal_counts)),
        "minimum_adjacent_bins": int(min(adjacent_counts)),
    }


def _partial_path(directory: Path, index: int) -> Path:
    return directory / f"record_{index:03d}.npz"


def _partial_fingerprint(raw_path: Path, *, event: str, source_hash: str) -> str:
    stat = raw_path.stat()
    payload = {
        "partial_schema": PARTIAL_SCHEMA,
        "method_version": METHOD_VERSION,
        "event": event,
        "source_event_cache_sha256": source_hash,
        "raw_path": str(raw_path.resolve()),
        "raw_bytes": int(stat.st_size),
        "raw_mtime_ns": int(stat.st_mtime_ns),
        "raw_window_sizes": RAW_WINDOW_SIZES,
        "burst_feature_names": BURST_FEATURE_NAMES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_partial(
    path: Path,
    *,
    event: str,
    features: np.ndarray,
    window_start: np.ndarray,
    audit: dict[str, Any],
    fingerprint: str,
    seconds: float,
) -> None:
    _atomic_npz(
        path,
        {
            "event": np.asarray(event),
            "burst_features": np.asarray(features, dtype=np.float32),
            "window_start_sample": np.asarray(window_start, dtype=np.int64),
            "audit_json": np.asarray(json.dumps(audit, sort_keys=True)),
            "provenance_fingerprint": np.asarray(fingerprint),
            "partial_schema": np.asarray(PARTIAL_SCHEMA),
            "seconds": np.asarray(seconds, dtype=np.float64),
        },
    )


def _load_partial(
    path: Path,
    *,
    event: str,
    fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    with np.load(path, allow_pickle=False) as cache:
        if str(np.asarray(cache["partial_schema"]).item()) != PARTIAL_SCHEMA:
            raise RuntimeError(f"partial schema drift: {path}")
        if str(np.asarray(cache["event"]).item()) != event:
            raise RuntimeError(f"partial event drift: {path}")
        if str(np.asarray(cache["provenance_fingerprint"]).item()) != fingerprint:
            raise RuntimeError(f"partial provenance drift: {path}")
        features = np.asarray(cache["burst_features"], dtype=np.float32)
        window_start = np.asarray(cache["window_start_sample"], dtype=np.int64)
        audit = json.loads(str(np.asarray(cache["audit_json"]).item()))
        seconds = float(np.asarray(cache["seconds"]).item())
    if features.ndim != 2 or features.shape[1] != len(BURST_FEATURE_NAMES):
        raise RuntimeError(f"partial burst feature shape drift: {path}")
    if window_start.shape != (len(features),) or not np.isfinite(features).all():
        raise RuntimeError(f"partial payload is invalid: {path}")
    return features, window_start, audit, seconds


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = load_event_cache(args.event_cache)
    inventory = load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    if tuple(row.event for row in inventory) != tuple(source["event"].tolist()):
        raise RuntimeError("raw inventory and Detector-v2 cache order differ")
    processed_now = 0
    for record_index, row in enumerate(inventory):
        partial = _partial_path(args.partial_dir, record_index)
        raw_path = Path(row.path)
        fingerprint = _partial_fingerprint(
            raw_path, event=row.event, source_hash=source["cache_sha256"]
        )
        if partial.exists():
            _load_partial(partial, event=row.event, fingerprint=fingerprint)
            continue
        first = int(source["record_offsets"][record_index])
        stop = int(source["record_offsets"][record_index + 1])
        record_started = time.perf_counter()
        with h5py.File(raw_path, "r") as handle:
            features, window_start, audit = extract_record_slow_time(
                handle["RF0_I"],
                handle["RF0_Q"],
                source["event_features"][first:stop],
                source["event_feature_names"],
                sample_rate_hz=source["sample_rate_hz"],
                hop_size=source["hop_size"],
                recording_samples=source["recording_samples"],
            )
        elapsed = time.perf_counter() - record_started
        _write_partial(
            partial,
            event=row.event,
            features=features,
            window_start=window_start,
            audit=audit,
            fingerprint=fingerprint,
            seconds=elapsed,
        )
        processed_now += 1
        print(
            f"slow-time {record_index + 1}/{len(inventory)} {row.event} "
            f"tf={audit['detector_tf_components']} "
            f"bursts={audit['physical_time_bursts']} seconds={elapsed:.2f}",
            flush=True,
        )
        if args.max_recordings and processed_now >= args.max_recordings:
            break

    completed = sum(
        _partial_path(args.partial_dir, index).exists() for index in range(len(inventory))
    )
    if completed != len(inventory):
        return {
            "status": "partial",
            "completed_recordings": completed,
            "remaining_recordings": len(inventory) - completed,
            "processed_now": processed_now,
            "partial_dir": str(args.partial_dir.resolve()),
            "seconds": float(time.perf_counter() - started),
        }

    rows: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    elapsed: list[float] = []
    offsets = [0]
    for index, row in enumerate(inventory):
        raw_path = Path(row.path)
        fingerprint = _partial_fingerprint(
            raw_path, event=row.event, source_hash=source["cache_sha256"]
        )
        feature, window_start, audit, seconds = _load_partial(
            _partial_path(args.partial_dir, index),
            event=row.event,
            fingerprint=fingerprint,
        )
        rows.append(feature)
        starts.append(window_start)
        audits.append(audit)
        elapsed.append(seconds)
        offsets.append(offsets[-1] + len(feature))
    burst_features = np.concatenate(rows, axis=0)
    window_start = np.concatenate(starts, axis=0)
    payload = {
        "event": source["event"],
        "identity": source["identity"],
        "class_index": source["class_index"],
        "condition": source["condition"],
        "class_names": np.asarray(BEHAVIOR5_CLASSES),
        "burst_feature_names": np.asarray(BURST_FEATURE_NAMES),
        "record_offsets": np.asarray(offsets, dtype=np.int64),
        "burst_features": burst_features,
        "window_start_sample": window_start,
        "sample_rate_hz": np.asarray(source["sample_rate_hz"], dtype=np.float64),
        "recording_samples": np.asarray(source["recording_samples"], dtype=np.int64),
        "raw_window_sizes": np.asarray(RAW_WINDOW_SIZES, dtype=np.int64),
        "schema": np.asarray(CACHE_SCHEMA),
        "method_version": np.asarray(METHOD_VERSION),
        "source_event_cache_sha256": np.asarray(source["cache_sha256"]),
    }
    _atomic_npz(args.output, payload)
    physical_counts = [float(row["physical_time_bursts"]) for row in audits]
    report = {
        "schema": REPORT_SCHEMA,
        "method_version": METHOD_VERSION,
        "status": "complete",
        "protocol": {
            "unit": "one complete 1.4 s recording",
            "downstream_outer_protocol": "strict eight-fold leave-one-UAV/model-out",
            "labels_or_metadata_seen_by_extractor": [],
            "condition_used": False,
            "target_cohort_statistics_used": False,
        },
        "representation": {
            "detector_tf_components": int(len(source["event_features"])),
            "physical_time_bursts": int(len(burst_features)),
            "merge_rule": "strict temporal interval overlap union on exact integer Detector-v2 frame ticks across frequency components",
            "timestamp": "exact representative-component midpoint converted to raw sample index within each temporal union",
            "exact_timestamp_storage": "int64 window_start_sample plus 1024-sample center at 100 MS/s",
            "sample_rate_hz": source["sample_rate_hz"],
            "raw_window_sizes": list(RAW_WINDOW_SIZES),
            "raw_complex_samples_read": int(len(burst_features) * max(RAW_WINDOW_SIZES)),
            "cross_burst_slow_time_observed": True,
            "burst_internal_spectral_shape_used": False,
            "energy": "mean signal-band power minus matched adjacent-band mean power in dB/bin",
            "absolute_power_retained_for_downstream": False,
        },
        "audit": {
            "physical_bursts_per_recording": _distribution(physical_counts),
            "seconds_per_recording": _distribution(elapsed),
            "minimum_signal_bins": int(min(row["minimum_signal_bins"] for row in audits)),
            "minimum_adjacent_bins": int(min(row["minimum_adjacent_bins"] for row in audits)),
            "recordings_with_at_least_three_bursts": int(
                sum(row["physical_time_bursts"] >= 3 for row in audits)
            ),
        },
        "source_event_cache": {
            "path": str(args.event_cache.resolve()),
            "sha256": source["cache_sha256"],
            "complete_marker": source["marker"],
        },
        "cache": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "bytes": int(args.output.stat().st_size),
            "schema": CACHE_SCHEMA,
            "burst_feature_shape": list(burst_features.shape),
        },
    }
    _atomic_json(args.report, report)
    marker = args.output.with_suffix(args.output.suffix + ".complete.json")
    _atomic_json(
        marker,
        {
            "status": "complete",
            "schema": CACHE_SCHEMA,
            "method_version": METHOD_VERSION,
            "cache": str(args.output.resolve()),
            "cache_sha256": _sha256(args.output),
            "report": str(args.report.resolve()),
            "report_sha256": _sha256(args.report),
            "source_event_cache_sha256": source["cache_sha256"],
        },
    )
    return {
        "status": "complete",
        "cache": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "physical_time_bursts": int(len(burst_features)),
        "processed_now": processed_now,
        "seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", nargs="?", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--event-cache", type=Path, default=DEFAULT_EVENT_CACHE)
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-recordings", type=int, default=0)
    args = parser.parse_args()
    if args.max_recordings < 0:
        parser.error("--max-recordings cannot be negative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
