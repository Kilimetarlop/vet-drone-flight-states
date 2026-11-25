"""Slow-time motion features of every recording, the inputs of the horizontal expert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

from .dataset import BEHAVIOR5_CLASSES, DATASET1_IDENTITIES
from .slow_time_motion import COMBINED_FEATURE_NAMES, LOW_FEATURE_NAMES, RATE_FEATURE_NAMES, RAW_WINDOW_SIZES, VIBRATION_FEATURE_NAMES, low_frequency_features, median_gap_nyquist_proxy_hz, q90_gap_nyquist_proxy_hz, rate_only_features, recording_local_residualize, vibration_features


METHOD_VERSION = "drff-r2-behavior5-cross-burst-slow-time-strict-lomo/v3"


INPUT_SCHEMA = "drff-r2-dataset1-behavior5-slow-time-energy/v3"


EXPECTED_RECORDINGS = 168


EXPECTED_ROWS_PER_UNIT = 21


EXPECTED_CLASS_COUNTS = (48, 48, 24, 24, 24)


SHUFFLE_SEED = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_slow_time_cache(path: Path) -> dict[str, Any]:
    marker = path.with_suffix(path.suffix + ".complete.json")
    if not marker.is_file():
        raise RuntimeError(f"slow-time complete marker is missing: {marker}")
    commit = json.loads(marker.read_text(encoding="utf-8"))
    observed_hash = _sha256(path)
    if (
        commit.get("status") != "complete"
        or commit.get("schema") != INPUT_SCHEMA
        or commit.get("cache_sha256") != observed_hash
    ):
        raise RuntimeError("slow-time cache complete marker or hash is invalid")
    required = {
        "event",
        "identity",
        "class_index",
        "condition",
        "class_names",
        "burst_feature_names",
        "record_offsets",
        "burst_features",
        "window_start_sample",
        "sample_rate_hz",
        "recording_samples",
        "schema",
    }
    with np.load(path, allow_pickle=False) as cache:
        missing = required - set(cache.files)
        if missing:
            raise RuntimeError(f"slow-time cache lacks keys: {sorted(missing)}")
        if str(np.asarray(cache["schema"]).item()) != INPUT_SCHEMA:
            raise RuntimeError("slow-time input schema drift")
        result = {
            "event": np.asarray(cache["event"]).astype(str),
            "identity": np.asarray(cache["identity"]).astype(str),
            "target": np.asarray(cache["class_index"], dtype=np.int64),
            "condition": np.asarray(cache["condition"], dtype=np.int64),
            "class_names": tuple(np.asarray(cache["class_names"]).astype(str).tolist()),
            "burst_feature_names": tuple(
                np.asarray(cache["burst_feature_names"]).astype(str).tolist()
            ),
            "record_offsets": np.asarray(cache["record_offsets"], dtype=np.int64),
            "burst_features": np.asarray(cache["burst_features"], dtype=np.float32),
            "window_start_sample": np.asarray(cache["window_start_sample"], dtype=np.int64),
            "sample_rate_hz": float(np.asarray(cache["sample_rate_hz"]).item()),
            "recording_samples": int(np.asarray(cache["recording_samples"]).item()),
            "cache_sha256": observed_hash,
        }
    if result["event"].shape != (EXPECTED_RECORDINGS,) or len(set(result["event"])) != EXPECTED_RECORDINGS:
        raise RuntimeError("expected 168 unique complete-recording rows")
    if result["identity"].shape != result["event"].shape or result["target"].shape != result["event"].shape:
        raise RuntimeError("record metadata vectors are not aligned")
    if result["class_names"] != BEHAVIOR5_CLASSES:
        raise RuntimeError("behavior class order drift")
    offsets = result["record_offsets"]
    if offsets.shape != (EXPECTED_RECORDINGS + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 3):
        raise RuntimeError("each recording must contain at least three physical bursts")
    if offsets[-1] != len(result["burst_features"]):
        raise RuntimeError("slow-time CSR payload is incomplete")
    if result["window_start_sample"].shape != (offsets[-1],):
        raise RuntimeError("raw-window start audit is not aligned")
    if not np.isfinite(result["burst_features"]).all():
        raise RuntimeError("slow-time burst features contain non-finite values")
    groups, counts = np.unique(result["identity"], return_counts=True)
    if tuple(sorted(groups.tolist())) != tuple(sorted(DATASET1_IDENTITIES)) or not np.all(counts == EXPECTED_ROWS_PER_UNIT):
        raise RuntimeError("strict eight-group inventory drift")
    class_counts = tuple(int(np.sum(result["target"] == index)) for index in range(5))
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(f"five-class count drift: {class_counts}")
    return result


def _shuffle_permutation(event: str, count: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{event}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(local_seed).permutation(count)


def build_record_features(source: dict[str, Any]) -> dict[str, np.ndarray]:
    names = source["burst_feature_names"]
    index = {name: names.index(name) for name in names}
    required = {
        "timestamp_seconds",
        "center_offset_hz",
        "occupied_bandwidth_hz",
        "band_excess_db_n1024",
        "band_excess_db_n2048",
    }
    if not required <= set(names):
        raise RuntimeError(f"slow-time cache lacks downstream channels: {sorted(required - set(names))}")
    duration = source["recording_samples"] / source["sample_rate_hz"]
    b0_rows: list[np.ndarray] = []
    b1_rows: list[np.ndarray] = []
    rate_rows: list[np.ndarray] = []
    shuffled_b0_rows: list[np.ndarray] = []
    shuffled_b1_rows: list[np.ndarray] = []
    median_gap_nyquist: list[float] = []
    q90_gap_nyquist: list[float] = []
    burst_count: list[int] = []
    timestamp_span: list[float] = []
    for record_index, event in enumerate(source["event"]):
        first = int(source["record_offsets"][record_index])
        stop = int(source["record_offsets"][record_index + 1])
        rows = source["burst_features"][first:stop]
        # The float32 timestamp channel is human-readable only.  Reconstruct
        # the measurement time from the exact int64 raw-window start so the
        # irregular spectrum never loses 100 MS/s sample precision.
        timestamp_sample = (
            source["window_start_sample"][first:stop]
            + max(RAW_WINDOW_SIZES) // 2
        )
        timestamp = timestamp_sample.astype(np.float64) / source["sample_rate_hz"]
        center = rows[:, index["center_offset_hz"]].astype(np.float64)
        bandwidth = rows[:, index["occupied_bandwidth_hz"]].astype(np.float64)
        raw_energy = np.column_stack(
            (
                rows[:, index["band_excess_db_n1024"]],
                rows[:, index["band_excess_db_n2048"]],
                0.5
                * (
                    rows[:, index["band_excess_db_n1024"]]
                    + rows[:, index["band_excess_db_n2048"]]
                ),
            )
        )
        residual = recording_local_residualize(
            raw_energy,
            center,
            bandwidth,
            sample_rate_hz=source["sample_rate_hz"],
        )
        b0_rows.append(
            low_frequency_features(timestamp, residual, duration_seconds=duration)
        )
        b1_rows.append(vibration_features(timestamp, residual))
        rate_rows.append(rate_only_features(timestamp, duration_seconds=duration))
        permutation = _shuffle_permutation(str(event), len(timestamp), SHUFFLE_SEED)
        shuffled = residual[permutation]
        shuffled_b0_rows.append(
            low_frequency_features(timestamp, shuffled, duration_seconds=duration)
        )
        shuffled_b1_rows.append(vibration_features(timestamp, shuffled))
        median_gap_nyquist.append(median_gap_nyquist_proxy_hz(timestamp))
        q90_gap_nyquist.append(q90_gap_nyquist_proxy_hz(timestamp))
        burst_count.append(len(timestamp))
        timestamp_span.append(float(timestamp[-1] - timestamp[0]))

    b0 = np.stack(b0_rows).astype(np.float32)
    b1 = np.stack(b1_rows).astype(np.float32)
    rate = np.stack(rate_rows).astype(np.float32)
    shuffled_b0 = np.stack(shuffled_b0_rows).astype(np.float32)
    shuffled_b1 = np.stack(shuffled_b1_rows).astype(np.float32)
    result = {
        "B0_low_frequency": b0,
        "B1_irregular_vibration": b1,
        "B2_multiresolution": np.concatenate((b0, b1), axis=1),
        "control_rate_only": rate,
        "control_time_shuffle_B0": shuffled_b0,
        "control_time_shuffle_B1": shuffled_b1,
        "control_time_shuffle_B2": np.concatenate((shuffled_b0, shuffled_b1), axis=1),
        "median_gap_nyquist_proxy_hz": np.asarray(
            median_gap_nyquist, dtype=np.float32
        ),
        "q90_gap_nyquist_proxy_hz": np.asarray(q90_gap_nyquist, dtype=np.float32),
        "physical_burst_count": np.asarray(burst_count, dtype=np.int64),
        "timestamp_span_seconds": np.asarray(timestamp_span, dtype=np.float32),
    }
    expected_width = {
        "B0_low_frequency": len(LOW_FEATURE_NAMES),
        "B1_irregular_vibration": len(VIBRATION_FEATURE_NAMES),
        "B2_multiresolution": len(COMBINED_FEATURE_NAMES),
        "control_rate_only": len(RATE_FEATURE_NAMES),
        "control_time_shuffle_B0": len(LOW_FEATURE_NAMES),
        "control_time_shuffle_B1": len(VIBRATION_FEATURE_NAMES),
        "control_time_shuffle_B2": len(COMBINED_FEATURE_NAMES),
    }
    for key, width in expected_width.items():
        if result[key].shape != (EXPECTED_RECORDINGS, width) or not np.isfinite(result[key]).all():
            raise RuntimeError(f"record feature contract drift: {key}={result[key].shape}")
    return result
