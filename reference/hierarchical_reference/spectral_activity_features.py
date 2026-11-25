"""454 summary features of the spectral-activity tracks.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from .build_spectral_activity import EVENTS, SCHEMA as TRAJECTORY_SCHEMA, SUBBANDS, TOKENS, TOKEN_FEATURE_NAMES
from .spectral_candidates import _burst_features


FEATURE_SCHEMA = "drff-r2-dataset1-behavior5-spectral-activity-features/v1"


RELATIVE_BAND_NAMES = TOKEN_FEATURE_NAMES[:SUBBANDS]


SCALAR_NAMES = TOKEN_FEATURE_NAMES[SUBBANDS:]


TEMPORAL_NAMES = (
    "log_total_power",
    "occupied_bandwidth_90",
    "spectral_entropy",
    "spectral_centroid",
    "peak_band_position",
    "hop_rate",
    "dwell_max_ratio",
    "spectral_flux_mean",
)


def _summary5(value: np.ndarray) -> tuple[float, ...]:
    quantile = np.quantile(value, (0.10, 0.50, 0.90))
    return (
        float(value.mean()), float(value.std()),
        float(quantile[0]), float(quantile[1]), float(quantile[2]),
    )


def _compact_periodic(value: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    centered = np.asarray(value, dtype=np.float64) - np.median(value)
    scale = max(1.4826 * np.median(np.abs(centered)), 1.0e-6)
    centered /= scale
    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocorrelation = np.fft.irfft(np.square(np.abs(spectrum)), n=fft_size)[: len(centered)]
    autocorrelation /= max(float(autocorrelation[0]), 1.0e-12)
    lags = (1, 2, 4, 8, 16, 32, 64, 128)
    feature = [float(autocorrelation[lag]) for lag in lags]
    names = [f"acf_lag_{lag}" for lag in lags]
    power = np.square(np.abs(np.fft.rfft(centered)))[1:]
    total = max(float(power.sum()), 1.0e-12)
    edges = np.unique(
        np.clip(
            np.rint(np.geomspace(1, len(power) + 1, 9)).astype(int),
            1,
            len(power) + 1,
        )
    )
    if len(edges) != 9:
        raise RuntimeError("compact periodic-band edge collapse")
    for index, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
        feature.append(float(power[first - 1 : last - 1].sum() / total))
        names.append(f"period_log_band_{index:02d}")
    probability = power / total
    feature.extend(
        (
            -float(np.sum(probability * np.log(np.maximum(probability, 1.0e-20))))
            / np.log(len(power)),
            float((np.argmax(power) + 1) / len(value)),
            float(power.max() / total),
        )
    )
    names.extend(("period_entropy", "dominant_frequency", "dominant_power_ratio"))
    return np.asarray(feature, dtype=np.float64), tuple(names)


def _trajectory_hop_features(
    peak_position: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    peak = np.clip(
        np.rint((peak_position + 1.0) * 0.5 * (SUBBANDS - 1)),
        0,
        SUBBANDS - 1,
    ).astype(np.int64)
    signed = np.diff(peak)
    jump = np.abs(signed).astype(np.float64)
    edges = np.flatnonzero(np.r_[True, peak[1:] != peak[:-1], True])
    runs = np.diff(edges).astype(np.float64)
    nonzero_signed = signed[signed != 0]
    reversals = 0.0
    if len(nonzero_signed) > 1:
        reversals = float(np.mean(nonzero_signed[1:] * nonzero_signed[:-1] < 0))
    values = (
        np.mean(jump > 0),
        jump.mean() / (SUBBANDS - 1),
        np.quantile(jump, 0.90) / (SUBBANDS - 1),
        jump.max() / (SUBBANDS - 1),
        len(np.unique(peak)) / SUBBANDS,
        len(runs) / len(peak),
        runs.mean() / len(peak),
        np.quantile(runs, 0.90) / len(peak),
        runs.max() / len(peak),
        reversals,
    )
    names = (
        "trajectory_hop_rate", "trajectory_hop_mean", "trajectory_hop_q90",
        "trajectory_hop_max", "trajectory_unique_band_fraction",
        "trajectory_dwell_count_rate", "trajectory_dwell_mean_ratio",
        "trajectory_dwell_q90_ratio", "trajectory_dwell_max_ratio",
        "trajectory_direction_reversal_rate",
    )
    return np.asarray(values, dtype=np.float64), names


def spectral_trajectory_feature_vector(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct one compact label-free event descriptor from all 1,424 tokens."""
    value = np.asarray(trajectory, dtype=np.float64)
    if value.shape != (TOKENS, len(TOKEN_FEATURE_NAMES)):
        raise ValueError(f"unexpected spectral trajectory shape: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("spectral trajectory contains non-finite values")
    index = {name: offset for offset, name in enumerate(TOKEN_FEATURE_NAMES)}
    features: list[float] = []
    names: list[str] = []

    for band_index, band_name in enumerate(RELATIVE_BAND_NAMES):
        series = value[:, band_index]
        quantile = np.quantile(series, (0.10, 0.90))
        features.extend((series.mean(), series.std(), *quantile))
        names.extend(
            f"{band_name}_{suffix}" for suffix in ("mean", "std", "q10", "q90")
        )
        absolute_delta = np.abs(np.diff(series))
        features.extend((absolute_delta.mean(), np.quantile(absolute_delta, 0.90)))
        names.extend((f"{band_name}_delta_abs_mean", f"{band_name}_delta_abs_q90"))

    for scalar_name in SCALAR_NAMES:
        features.extend(_summary5(value[:, index[scalar_name]]))
        names.extend(
            f"{scalar_name}_{suffix}"
            for suffix in ("mean", "std", "q10", "q50", "q90")
        )

    edges = np.linspace(0, TOKENS, 9, dtype=int)
    for bin_index, (first, last) in enumerate(zip(edges[:-1], edges[1:])):
        for scalar_name in TEMPORAL_NAMES:
            features.append(float(value[first:last, index[scalar_name]].mean()))
            names.append(f"temporal_bin_{bin_index}_{scalar_name}_mean")

    burst, burst_names = _burst_features(value[:, index["log_total_power"]])
    features.extend(burst.tolist())
    names.extend(burst_names)
    for series_name in ("log_total_power", "peak_band_position"):
        periodic, periodic_names = _compact_periodic(value[:, index[series_name]])
        features.extend(periodic.tolist())
        names.extend(f"{series_name}_{name}" for name in periodic_names)
    hop, hop_names = _trajectory_hop_features(value[:, index["peak_band_position"]])
    features.extend(hop.tolist())
    names.extend(hop_names)

    result = np.asarray(features, dtype=np.float32)
    if len(result) >= 500 or len(result) != len(names) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid spectral event feature vector: {result.shape}")
    return result, tuple(names)


def _record_power_robust(trajectory: np.ndarray) -> np.ndarray:
    result = np.asarray(trajectory, dtype=np.float32).copy()
    power_index = TOKEN_FEATURE_NAMES.index("log_total_power")
    power = result[:, power_index]
    median = float(np.median(power))
    scale = max(1.4826 * float(np.median(np.abs(power - median))), 1.0e-5)
    result[:, power_index] = np.clip((power - median) / scale, -50.0, 50.0)
    return result


def load_or_build_feature_cache(
    trajectory_path: Path, feature_path: Path,
) -> dict[str, np.ndarray]:
    with np.load(trajectory_path, allow_pickle=False) as cache:
        required = {"event", "identity", "class_index", "class_names", "schema", "trajectory"}
        missing = sorted(required - set(cache.files))
        if missing:
            raise RuntimeError(f"spectral trajectory cache missing keys: {missing}")
        if str(cache["schema"]) != TRAJECTORY_SCHEMA:
            raise RuntimeError("spectral trajectory schema mismatch")
        metadata = {
            "event": cache["event"].astype(str),
            "identity": cache["identity"].astype(str),
            "target": np.asarray(cache["class_index"], dtype=np.int64),
            "class_names": cache["class_names"].astype(str),
        }
        trajectory = np.asarray(cache["trajectory"], dtype=np.float32)
    expected = (EVENTS, TOKENS, len(TOKEN_FEATURE_NAMES))
    if trajectory.shape != expected or not np.isfinite(trajectory).all():
        raise RuntimeError(f"invalid spectral trajectory cache: {trajectory.shape}")
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as cache:
            if str(cache["schema"]) != FEATURE_SCHEMA:
                raise RuntimeError("spectral feature-cache schema mismatch")
            if not np.array_equal(cache["event"].astype(str), metadata["event"]):
                raise RuntimeError("spectral feature-cache event order mismatch")
            return {
                **metadata,
                "feature_names": cache["feature_names"].astype(str),
                "raw": np.asarray(cache["raw"], dtype=np.float32),
                "record_robust": np.asarray(cache["record_robust"], dtype=np.float32),
            }
    raw_rows, robust_rows = [], []
    names: tuple[str, ...] | None = None
    for event_index, event_trajectory in enumerate(trajectory):
        raw, current_names = spectral_trajectory_feature_vector(event_trajectory)
        robust, robust_names = spectral_trajectory_feature_vector(
            _record_power_robust(event_trajectory),
        )
        if current_names != robust_names or (names is not None and current_names != names):
            raise RuntimeError("spectral feature-name drift")
        names = current_names
        raw_rows.append(raw)
        robust_rows.append(robust)
        if (event_index + 1) % 20 == 0:
            print(f"spectral event features {event_index + 1}/{EVENTS}", flush=True)
    raw_matrix, robust_matrix = np.stack(raw_rows), np.stack(robust_rows)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_name(f"{feature_path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, schema=np.asarray(FEATURE_SCHEMA), event=metadata["event"],
            feature_names=np.asarray(names), raw=raw_matrix, record_robust=robust_matrix,
        )
    os.replace(temporary, feature_path)
    return {
        **metadata, "feature_names": np.asarray(names),
        "raw": raw_matrix, "record_robust": robust_matrix,
    }
