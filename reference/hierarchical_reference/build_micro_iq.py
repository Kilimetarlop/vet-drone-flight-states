"""Builds the micro-IQ cache: active windows, 64 uniform windows, and chronological probes of every
recording.
"""

from __future__ import annotations

from .dataset import DEFAULT_DATASET_ROOT as _DATASET_ROOT  # set DRFF_R2_DATASET1 to the Dataset 1 folder

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence
import h5py
import numpy as np

from .dataset import BEHAVIOR5_CLASSES, DRFFR2BehaviorRecording, EXPECTED_BEHAVIOR5_CLASS_COUNTS, load_drff_r2_dataset1_behavior5_inventory
from .micro_iq_extract import _extract_event


EVENTS = 168


WINDOWS = 29


SAMPLES = 16_384


CANDIDATES = 256


UNIFORM_WINDOWS = 64


PROBE_SAMPLES = 8_192


PROBE_SUBBANDS = 16


PROBE_FEATURE_NAMES = (
    "log_rms",
    *(f"log_subband_power_{index:02d}" for index in range(PROBE_SUBBANDS)),
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "phase_increment_coherence",
)


PROBE_FEATURES = len(PROBE_FEATURE_NAMES)


SCHEMA = "drff-r2-dataset1-behavior5-micro-iq-f16/v1"


REQUIRED_CACHE_KEYS = (
    "event",
    "identity",
    "behavior",
    "class_index",
    "condition",
    "selected_candidate_indices",
    "selected_centers",
    "window_starts",
    "scan_scores",
    "iq_real_imag",
    "uniform_window_starts",
    "uniform_iq_real_imag",
    "probe_window_starts",
    "probe_features",
    "probe_feature_names",
)


def _metadata(rows: Sequence[DRFFR2BehaviorRecording]) -> dict[str, np.ndarray]:
    if len(rows) != EVENTS:
        raise RuntimeError(f"behavior5 inventory drift: expected {EVENTS}, got {len(rows)}")
    metadata = {
        "event": np.asarray([row.event for row in rows]),
        "identity": np.asarray([row.identity for row in rows]),
        "behavior": np.asarray([row.behavior for row in rows]),
        "class_index": np.asarray([row.class_index for row in rows], dtype=np.int64),
        "condition": np.asarray([row.condition for row in rows], dtype=np.int64),
        "source_state": np.asarray([row.source_state for row in rows]),
        "channel": np.asarray([row.channel for row in rows]),
        "schema": np.asarray(SCHEMA),
        "probe_feature_names": np.asarray(PROBE_FEATURE_NAMES),
    }
    if tuple(sorted(set(metadata["behavior"].astype(str)))) != tuple(
        sorted(BEHAVIOR5_CLASSES)
    ):
        raise RuntimeError("behavior5 label inventory drift")
    if set(metadata["class_index"].tolist()) != set(range(5)):
        raise RuntimeError("behavior5 class-index inventory drift")
    return metadata


def _open_partial(
    partial_dir: Path,
    metadata: dict[str, np.ndarray],
) -> tuple[np.memmap, ...]:
    partial_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in metadata.items():
        path = partial_dir / f"{name}.npy"
        if not path.exists():
            np.save(path, expected, allow_pickle=False)
        observed = np.load(path, allow_pickle=False)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"partial metadata mismatch: {name}")

    specifications = {
        "selected_candidate_indices": (np.int64, (EVENTS, WINDOWS), -1),
        "selected_centers": (np.int64, (EVENTS, WINDOWS), -1),
        "window_starts": (np.int64, (EVENTS, WINDOWS), -1),
        "scan_scores": (np.float32, (EVENTS, CANDIDATES), np.nan),
        "iq_real_imag": (np.float16, (EVENTS, WINDOWS, SAMPLES, 2), 0.0),
        "uniform_window_starts": (
            np.int64,
            (EVENTS, UNIFORM_WINDOWS),
            -1,
        ),
        "uniform_iq_real_imag": (
            np.float16,
            (EVENTS, UNIFORM_WINDOWS, SAMPLES, 2),
            0.0,
        ),
        "probe_window_starts": (np.int64, (EVENTS, CANDIDATES), -1),
        "probe_features": (
            np.float32,
            (EVENTS, CANDIDATES, PROBE_FEATURES),
            np.nan,
        ),
        "completed": (np.bool_, (EVENTS,), False),
    }
    arrays: list[np.memmap] = []
    for name, (dtype, shape, fill) in specifications.items():
        path = partial_dir / f"{name}.npy"
        if not path.exists():
            value = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            value[:] = fill
            value.flush()
            del value
        value = np.lib.format.open_memmap(path, mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"partial array drift: {name} {value.shape} {value.dtype}"
            )
        arrays.append(value)
    return tuple(arrays)


def _validate_extracted_event(values: Sequence[np.ndarray]) -> None:
    expected = (
        (np.dtype(np.int64), (WINDOWS,)),
        (np.dtype(np.int64), (WINDOWS,)),
        (np.dtype(np.int64), (WINDOWS,)),
        (np.dtype(np.float32), (CANDIDATES,)),
        (np.dtype(np.float16), (WINDOWS, SAMPLES, 2)),
    )
    if len(values) != len(expected):
        raise RuntimeError("_extract_event output arity drift")
    for index, (value, (dtype, shape)) in enumerate(zip(values, expected, strict=True)):
        array = np.asarray(value)
        if array.shape != shape or array.dtype != dtype:
            raise RuntimeError(
                f"_extract_event output {index} drift: {array.shape} {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"_extract_event output {index} contains non-finite data")


def _probe_feature(value: np.ndarray, taper: np.ndarray) -> np.ndarray:
    """Summarize one chronological 8192-sample probe without label access."""

    centered = np.asarray(value, dtype=np.complex64)
    centered = centered - centered.mean()
    rms = float(np.sqrt(np.mean(np.abs(centered) ** 2, dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("behavior probe contains degenerate I/Q")

    # complex128: NumPy 2 transforms complex64 input in single precision
    spectrum = np.fft.fftshift(np.fft.fft((centered * taper).astype(np.complex128)))
    power = np.square(np.abs(spectrum), dtype=np.float64)
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("behavior probe contains degenerate spectrum")
    probability = power / total
    subband_power = power.reshape(PROBE_SUBBANDS, -1).mean(axis=1)
    eps = np.finfo(np.float64).tiny
    frequency = np.linspace(-1.0, 1.0, PROBE_SAMPLES, endpoint=False)
    centroid = float(np.sum(frequency * probability))
    bandwidth = float(
        np.sqrt(np.sum(np.square(frequency - centroid) * probability))
    )
    entropy = float(
        -np.sum(probability * np.log(probability + eps)) / np.log(PROBE_SAMPLES)
    )
    flatness = float(
        np.exp(np.mean(np.log(power + eps))) / max(float(power.mean()), eps)
    )
    phase_product = centered[1:] * np.conj(centered[:-1])
    phase_denominator = np.abs(centered[1:]) * np.abs(centered[:-1])
    valid = phase_denominator > np.finfo(np.float32).eps
    if bool(valid.any()):
        coherence = float(
            np.abs(np.mean(phase_product[valid] / phase_denominator[valid]))
        )
    else:
        coherence = 0.0
    result = np.asarray(
        (
            np.log(rms + eps),
            *np.log(subband_power + eps).tolist(),
            entropy,
            centroid,
            bandwidth,
            flatness,
            coherence,
        ),
        dtype=np.float32,
    )
    if result.shape != (PROBE_FEATURES,) or not np.isfinite(result).all():
        raise RuntimeError("invalid behavior probe feature vector")
    return result


def _extract_behavior_views(path: Path) -> tuple[np.ndarray, ...]:
    """Extract full-span ordered windows and dense chronological probe features."""

    taper = np.hanning(PROBE_SAMPLES).astype(np.float32)
    with h5py.File(path, "r") as source:
        if "RF0_I" not in source or "RF0_Q" not in source:
            raise RuntimeError(f"missing RF0_I/RF0_Q: {path.name}")
        i_data, q_data = source["RF0_I"], source["RF0_Q"]
        length = int(i_data.shape[0])
        if q_data.shape != i_data.shape or length < SAMPLES:
            raise RuntimeError(f"invalid RF0_I/RF0_Q: {path.name}")

        uniform_starts = np.linspace(
            0,
            length - SAMPLES,
            UNIFORM_WINDOWS,
            dtype=np.int64,
        )
        if (
            uniform_starts[0] != 0
            or uniform_starts[-1] != length - SAMPLES
            or not bool(np.all(np.diff(uniform_starts) > 0))
        ):
            raise RuntimeError("uniform behavior windows do not span the recording")
        uniform_iq = np.empty(
            (UNIFORM_WINDOWS, SAMPLES, 2),
            dtype=np.float16,
        )
        for row, start in enumerate(uniform_starts.tolist()):
            real = np.asarray(i_data[start : start + SAMPLES, 0], dtype=np.float32)
            imag = np.asarray(q_data[start : start + SAMPLES, 0], dtype=np.float32)
            value = np.stack((real, imag), axis=1)
            if value.shape != (SAMPLES, 2) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid uniform micro-IQ: {path.name}/{row}")
            if float(np.max(np.abs(value))) > np.finfo(np.float16).max:
                raise RuntimeError(f"uniform float16 overflow: {path.name}/{row}")
            uniform_iq[row] = value.astype(np.float16)

        probe_centers = np.linspace(
            PROBE_SAMPLES // 2,
            length - PROBE_SAMPLES // 2,
            CANDIDATES,
            dtype=np.int64,
        )
        probe_starts = probe_centers - PROBE_SAMPLES // 2
        probe_features = np.empty(
            (CANDIDATES, PROBE_FEATURES),
            dtype=np.float32,
        )
        for row, start in enumerate(probe_starts.tolist()):
            real = np.asarray(
                i_data[start : start + PROBE_SAMPLES, 0],
                dtype=np.float32,
            )
            imag = np.asarray(
                q_data[start : start + PROBE_SAMPLES, 0],
                dtype=np.float32,
            )
            if real.shape != (PROBE_SAMPLES,) or imag.shape != (PROBE_SAMPLES,):
                raise RuntimeError(f"invalid chronological probe: {path.name}/{row}")
            probe_features[row] = _probe_feature(real + 1j * imag, taper)
    return uniform_starts, uniform_iq, probe_starts, probe_features


def _validate_existing_cache(
    output: Path,
    metadata: dict[str, np.ndarray],
) -> None:
    with np.load(output, allow_pickle=False) as cache:
        missing = sorted(set(REQUIRED_CACHE_KEYS) - set(cache.files))
        if missing:
            raise RuntimeError(f"existing behavior5 cache lacks keys: {missing}")
        for name, expected in metadata.items():
            if name not in cache.files or not np.array_equal(cache[name], expected):
                raise RuntimeError(f"existing behavior5 cache metadata mismatch: {name}")
        expected_shapes = {
            "selected_candidate_indices": (EVENTS, WINDOWS),
            "selected_centers": (EVENTS, WINDOWS),
            "window_starts": (EVENTS, WINDOWS),
            "scan_scores": (EVENTS, CANDIDATES),
            "iq_real_imag": (EVENTS, WINDOWS, SAMPLES, 2),
            "uniform_window_starts": (EVENTS, UNIFORM_WINDOWS),
            "uniform_iq_real_imag": (
                EVENTS,
                UNIFORM_WINDOWS,
                SAMPLES,
                2,
            ),
            "probe_window_starts": (EVENTS, CANDIDATES),
            "probe_features": (EVENTS, CANDIDATES, PROBE_FEATURES),
        }
        for name, shape in expected_shapes.items():
            if cache[name].shape != shape:
                raise RuntimeError(
                    f"existing behavior5 cache shape mismatch: {name}={cache[name].shape}"
                )
        if cache["iq_real_imag"].dtype != np.float16:
            raise RuntimeError("existing behavior5 cache IQ dtype mismatch")
        if cache["uniform_iq_real_imag"].dtype != np.float16:
            raise RuntimeError("existing behavior5 cache uniform IQ dtype mismatch")
        if not np.array_equal(
            cache["probe_feature_names"].astype(str),
            np.asarray(PROBE_FEATURE_NAMES),
        ):
            raise RuntimeError("existing behavior5 cache probe feature-name mismatch")


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    rows = load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    metadata = _metadata(rows)
    if args.output_cache.exists():
        _validate_existing_cache(args.output_cache, metadata)
        return {
            "status": "existing",
            "path": str(args.output_cache.resolve()),
            "events": EVENTS,
            "class_counts": EXPECTED_BEHAVIOR5_CLASS_COUNTS,
            "seconds": time.perf_counter() - started,
        }

    (
        selected_indices,
        selected_centers,
        window_starts,
        scan_scores,
        iq_real_imag,
        uniform_window_starts,
        uniform_iq_real_imag,
        probe_window_starts,
        probe_features,
        completed,
    ) = _open_partial(args.partial_dir, metadata)
    resumed = int(completed.sum())
    destinations = (
        selected_indices,
        selected_centers,
        window_starts,
        scan_scores,
        iq_real_imag,
    )
    for row_index, row in enumerate(rows):
        if bool(completed[row_index]):
            continue
        values = _extract_event(Path(row.path))
        _validate_extracted_event(values)
        for destination, value in zip(destinations, values, strict=True):
            destination[row_index] = value
            destination.flush()
        behavior_values = _extract_behavior_views(Path(row.path))
        for destination, value in zip(
            (
                uniform_window_starts,
                uniform_iq_real_imag,
                probe_window_starts,
                probe_features,
            ),
            behavior_values,
            strict=True,
        ):
            destination[row_index] = value
            destination.flush()
        completed[row_index] = True
        completed.flush()
        print(
            f"behavior5 micro-IQ {int(completed.sum())}/{EVENTS} {row.event} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not bool(completed.all()):
        raise RuntimeError("behavior5 extraction incomplete")

    audit = {
        "schema": SCHEMA,
        "classes": list(BEHAVIOR5_CLASSES),
        "class_counts": EXPECTED_BEHAVIOR5_CLASS_COUNTS,
        "source_recordings": 208,
        "selected_recordings": EVENTS,
        "excluded_conditions": [22, 23, 24, 25, 26],
        "active_branch": "activity-ranked 256-candidate scan with chronological 29-window NMS",
        "behavior_branch": (
            "64 strictly ordered uniform windows spanning each complete recording, "
            "plus 256 ordered 8192-sample probe descriptors"
        ),
        "uniform_windows": UNIFORM_WINDOWS,
        "probe_features": list(PROBE_FEATURE_NAMES),
    }
    payload = {
        **metadata,
        "selected_candidate_indices": np.asarray(selected_indices),
        "selected_centers": np.asarray(selected_centers),
        "window_starts": np.asarray(window_starts),
        "scan_scores": np.asarray(scan_scores),
        "iq_real_imag": np.asarray(iq_real_imag),
        "uniform_window_starts": np.asarray(uniform_window_starts),
        "uniform_iq_real_imag": np.asarray(uniform_iq_real_imag),
        "probe_window_starts": np.asarray(probe_window_starts),
        "probe_features": np.asarray(probe_features),
        "class_names": np.asarray(BEHAVIOR5_CLASSES),
        "audit_json": np.asarray(json.dumps(audit, sort_keys=True)),
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_cache.with_name(
        f"{args.output_cache.stem}.{os.getpid()}.tmp.npz"
    )
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, args.output_cache)
    _validate_existing_cache(args.output_cache, metadata)
    return {
        "status": "complete",
        "path": str(args.output_cache.resolve()),
        "shape": [EVENTS, WINDOWS, SAMPLES, 2],
        "dtype": "float16",
        "bytes": int(args.output_cache.stat().st_size),
        "events": EVENTS,
        "classes": list(BEHAVIOR5_CLASSES),
        "class_counts": EXPECTED_BEHAVIOR5_CLASS_COUNTS,
        "resumed_events": resumed,
        "extracted_events": EVENTS - resumed,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=_DATASET_ROOT,
    )
    parser.add_argument(
        "--partial-dir",
        type=Path,
        default=Path("tmp/drff_r2_dataset1_behavior5_micro_iq_f16.partial"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("tmp/drff_r2_dataset1_behavior5_micro_iq_f16.npz"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)
