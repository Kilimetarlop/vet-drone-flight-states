"""MultiRocket-Hydra branch on the spectral-activity tracks with a ridge classifier.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
import time
from typing import Any
import numpy as np
from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from .build_spectral_activity import SCHEMA as TRAJECTORY_SCHEMA, SUBBANDS, TOKENS, TOKEN_FEATURE_NAMES
from .dualtime import CLASS_NAMES


EXPECTED_EVENTS = 168


EXPECTED_GROUPS = 8


EXPECTED_EVENTS_PER_GROUP = 21


EXPECTED_CLASS_COUNTS = (48, 48, 24, 24, 24)


CENTER_BAND_INDEX = 16


ROBUST_CLIP = 8.0


MAD_NORMALIZER = 1.4826


ALPHAS = np.logspace(-3, 3, 7, dtype=np.float64)


class ProgressMonitor:
    """Track native RSS and emit a heartbeat while long transforms run."""

    def __init__(self, interval_seconds: float = 120.0) -> None:
        self.interval_seconds = float(interval_seconds)
        self.started = time.perf_counter()
        self.stage = "initialising"
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except ImportError:  # pragma: no cover - psutil is present in the runtime
            self._process = None
        self._windows_process = None
        self._windows_counters_type = None
        if self._process is None and os.name == "nt":
            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            self._windows_process = ctypes.windll.kernel32.GetCurrentProcess()
            self._windows_counters_type = ProcessMemoryCountersEx

    def _rss(self) -> int:
        value = 0
        if self._process is not None:
            value = int(self._process.memory_info().rss)
        elif self._windows_process is not None:
            counters = self._windows_counters_type()
            counters.cb = ctypes.sizeof(counters)
            success = ctypes.windll.psapi.GetProcessMemoryInfo(
                self._windows_process,
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                value = int(counters.WorkingSetSize)
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(counters.PeakWorkingSetSize),
                )
        self.peak_rss_bytes = max(self.peak_rss_bytes, value)
        return value

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)
        rss_mb = self._rss() / (1024**2)
        print(f"stage={self.stage} rss_mb={rss_mb:.1f}", flush=True)

    def _run(self) -> None:
        next_heartbeat = time.monotonic() + self.interval_seconds
        while not self._stop.wait(0.5):
            rss = self._rss()
            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = time.perf_counter() - self.started
                print(
                    f"heartbeat stage={self.stage} elapsed_s={elapsed:.1f} "
                    f"rss_mb={rss / (1024**2):.1f} "
                    f"peak_rss_mb={self.peak_rss_bytes / (1024**2):.1f}",
                    flush=True,
                )
                next_heartbeat = now + self.interval_seconds

    def __enter__(self) -> "ProgressMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_error: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._rss()


def _feature_index() -> dict[str, int]:
    return {str(name): index for index, name in enumerate(TOKEN_FEATURE_NAMES)}


def align_record_carrier_and_power(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply only label-free, single-record alignment.

    All tokens have the same duration, so the duration-weighted median of the
    peak position is exactly its temporal median.  The corresponding integer
    band is shifted to zero-based band 16.  Newly exposed spectral cells use the
    minimum of that token's 32 relative log-power bands.
    """
    value = np.asarray(trajectory, dtype=np.float32)
    expected = (TOKENS, len(TOKEN_FEATURE_NAMES))
    if value.shape != expected:
        raise ValueError(f"unexpected record trajectory shape: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("record trajectory contains non-finite values")
    result = value.copy()
    index = _feature_index()
    peak_index = index["peak_band_position"]
    centroid_index = index["spectral_centroid"]
    power_index = index["log_total_power"]

    center_position = float(np.median(value[:, peak_index]))
    center_band = int(
        np.clip(
            np.rint((center_position + 1.0) * 0.5 * (SUBBANDS - 1)),
            0,
            SUBBANDS - 1,
        )
    )
    shift = int(CENTER_BAND_INDEX - center_band)
    bands = value[:, :SUBBANDS]
    shifted = np.broadcast_to(
        bands.min(axis=1, keepdims=True), bands.shape,
    ).copy()
    if shift > 0:
        shifted[:, shift:] = bands[:, : SUBBANDS - shift]
    elif shift < 0:
        shifted[:, : SUBBANDS + shift] = bands[:, -shift:]
    else:
        shifted[:] = bands
    result[:, :SUBBANDS] = shifted

    # Coordinates are expressed relative to the record carrier, not the
    # absolute receiver spectrum.  This is synchronized with the band shift.
    result[:, peak_index] = np.clip(
        value[:, peak_index] - center_position, -1.0, 1.0,
    )
    result[:, centroid_index] = np.clip(
        value[:, centroid_index] - center_position, -1.0, 1.0,
    )
    power_median = float(np.median(value[:, power_index]))
    result[:, power_index] = value[:, power_index] - power_median
    return result, {
        "carrier_center_position": center_position,
        "carrier_center_band": center_band,
        "integer_band_shift": shift,
        "log_total_power_median": power_median,
    }


def align_all_records(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    value = np.asarray(trajectory, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (
        TOKENS,
        len(TOKEN_FEATURE_NAMES),
    ):
        raise ValueError(f"invalid trajectory collection: {value.shape}")
    aligned = np.empty_like(value)
    audits: list[dict[str, float | int]] = []
    for record_index in range(len(value)):
        aligned[record_index], audit = align_record_carrier_and_power(
            value[record_index],
        )
        audits.append(audit)
    return aligned, audits


def fit_source_robust_scaler(
    aligned_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel median/MAD on outer-source tokens only."""
    source = np.asarray(aligned_source, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (
        TOKENS,
        len(TOKEN_FEATURE_NAMES),
    ):
        raise ValueError(f"invalid source trajectory collection: {source.shape}")
    flattened = source.reshape(-1, source.shape[-1])
    center = np.median(flattened, axis=0).astype(np.float32)
    mad = np.median(np.abs(flattened - center), axis=0).astype(np.float32)
    scale = (MAD_NORMALIZER * mad).astype(np.float32)
    scale = np.maximum(scale, np.float32(1.0e-6))
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise RuntimeError("non-finite source robust-scaler parameters")
    return center, scale


def apply_source_robust_scaler(
    aligned: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    value = np.asarray(aligned, dtype=np.float32)
    result = (value - np.asarray(center, dtype=np.float32)[None, None, :]) / np.asarray(
        scale, dtype=np.float32,
    )[None, None, :]
    return np.clip(result, -ROBUST_CLIP, ROBUST_CLIP).transpose(0, 2, 1).astype(
        np.float32,
        copy=False,
    )


def make_fixed_transformers() -> tuple[MultiRocket, HydraTransformer]:
    return (
        MultiRocket(
            n_kernels=1250,
            max_dilations_per_kernel=32,
            n_features_per_kernel=4,
            normalise=False,
            n_jobs=-1,
            random_state=0,
        ),
        HydraTransformer(
            n_kernels=8,
            n_groups=64,
            max_num_channels=8,
            n_jobs=-1,
            random_state=0,
            output_type="numpy",
        ),
    )


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "event", "identity", "behavior", "class_index", "class_names",
            "token_feature_names", "schema", "trajectory",
        }
        missing = sorted(required - set(cache.files))
        if missing:
            raise RuntimeError(f"spectral cache missing keys: {missing}")
        if str(cache["schema"].item()) != TRAJECTORY_SCHEMA:
            raise RuntimeError("spectral trajectory schema mismatch")
        result = {key: np.asarray(cache[key]) for key in required}
    event = result["event"].astype(str)
    group = result["identity"].astype(str)
    behavior = result["behavior"].astype(str)
    target = np.asarray(result["class_index"], dtype=np.int64)
    trajectory = np.asarray(result["trajectory"], dtype=np.float32)
    class_names = result["class_names"].astype(str)
    feature_names = result["token_feature_names"].astype(str)
    if event.shape != (EXPECTED_EVENTS,) or len(set(event.tolist())) != EXPECTED_EVENTS:
        raise RuntimeError("expected 168 unique complete recordings")
    if trajectory.shape != (EXPECTED_EVENTS, TOKENS, len(TOKEN_FEATURE_NAMES)):
        raise RuntimeError(f"spectral trajectory shape mismatch: {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise RuntimeError("spectral cache contains non-finite values")
    if not np.array_equal(class_names, np.asarray(CLASS_NAMES)):
        raise RuntimeError("five-class name/order mismatch")
    if not np.array_equal(feature_names, np.asarray(TOKEN_FEATURE_NAMES)):
        raise RuntimeError("spectral token feature-name/order mismatch")
    if not np.array_equal(behavior, class_names[target]):
        raise RuntimeError("behavior and class index mismatch")
    counts = tuple(int(np.sum(target == index)) for index in range(len(CLASS_NAMES)))
    if counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(f"five-class count drift: {counts}")
    group_names, group_counts = np.unique(group, return_counts=True)
    if len(group_names) != EXPECTED_GROUPS or set(group_counts.tolist()) != {
        EXPECTED_EVENTS_PER_GROUP
    }:
        raise RuntimeError("expected eight groups with 21 recordings each")
    return {
        "event": event,
        "identity": group,
        "behavior": behavior,
        "target": target,
        "trajectory": trajectory,
    }


def _fit_transform_fold(
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_y: np.ndarray,
    monitor: ProgressMonitor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    timing: dict[str, float] = {}
    multirocket, hydra = make_fixed_transformers()

    monitor.set_stage("multirocket_fit")
    tick = time.perf_counter()
    multirocket.fit(source_x)
    timing["multirocket_fit_seconds"] = time.perf_counter() - tick
    monitor.set_stage("multirocket_transform_source")
    tick = time.perf_counter()
    mr_source = np.asarray(multirocket.transform(source_x), dtype=np.float32)
    timing["multirocket_transform_source_seconds"] = time.perf_counter() - tick
    monitor.set_stage("multirocket_transform_target")
    tick = time.perf_counter()
    mr_target = np.asarray(multirocket.transform(target_x), dtype=np.float32)
    timing["multirocket_transform_target_seconds"] = time.perf_counter() - tick

    monitor.set_stage("hydra_fit")
    tick = time.perf_counter()
    hydra.fit(source_x)
    timing["hydra_fit_seconds"] = time.perf_counter() - tick
    monitor.set_stage("hydra_transform_source")
    tick = time.perf_counter()
    hy_source = np.asarray(hydra.transform(source_x), dtype=np.float32)
    timing["hydra_transform_source_seconds"] = time.perf_counter() - tick
    monitor.set_stage("hydra_transform_target")
    tick = time.perf_counter()
    hy_target = np.asarray(hydra.transform(target_x), dtype=np.float32)
    timing["hydra_transform_target_seconds"] = time.perf_counter() - tick

    monitor.set_stage("branch_standardise_and_ridge_fit")
    tick = time.perf_counter()
    mr_scaler = StandardScaler()
    hy_scaler = StandardScaler()
    mr_source = mr_scaler.fit_transform(mr_source).astype(np.float32, copy=False)
    mr_target = mr_scaler.transform(mr_target).astype(np.float32, copy=False)
    hy_source = hy_scaler.fit_transform(hy_source).astype(np.float32, copy=False)
    hy_target = hy_scaler.transform(hy_target).astype(np.float32, copy=False)
    source_feature = np.concatenate((mr_source, hy_source), axis=1)
    target_feature = np.concatenate((mr_target, hy_target), axis=1)
    classifier = RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced")
    classifier.fit(source_feature, source_y)
    prediction = np.asarray(classifier.predict(target_feature), dtype=np.int64)
    decision = np.asarray(classifier.decision_function(target_feature), dtype=np.float32)
    timing["branch_scaling_and_ridge_seconds"] = time.perf_counter() - tick
    timing["fold_model_seconds"] = float(sum(timing.values()))
    dimensions = {
        "multirocket": int(mr_source.shape[1]),
        "hydra": int(hy_source.shape[1]),
        "combined": int(source_feature.shape[1]),
    }
    return prediction, decision, np.asarray(classifier.alpha_), {
        "timing_seconds": timing,
        "feature_dimensions": dimensions,
    }
