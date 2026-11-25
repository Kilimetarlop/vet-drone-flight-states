"""Slow-time energy and vibration features around detected bursts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


RAW_WINDOW_SIZES = (1_024, 2_048)


LOW_FREQUENCY_BANDS_HZ = (
    (0.7, 1.5),
    (1.5, 3.0),
    (3.0, 5.0),
    (5.0, 7.5),
    (7.5, 10.0),
)


VIBRATION_BANDS_HZ = (
    (10.0, 30.0),
    (30.0, 50.0),
    (50.0, 90.0),
    (90.0, 140.0),
    (140.0, 220.0),
)


ENERGY_CHANNEL_NAMES = (
    "band_excess_db_n1024",
    "band_excess_db_n2048",
    "multiresolution_mean_excess_db",
)


SCALE_FLOOR = 1.0e-9


@dataclass(frozen=True)
class PhysicalBurst:
    """One temporal union of overlapping detector TF components."""

    start_seconds: float
    stop_seconds: float
    representative_index: int
    component_count: int

    @property
    def midpoint_seconds(self) -> float:
        return 0.5 * (self.start_seconds + self.stop_seconds)

    @property
    def duration_seconds(self) -> float:
        return self.stop_seconds - self.start_seconds


def _feature_index(channel_names: Iterable[str]) -> dict[str, int]:
    names = tuple(str(name) for name in channel_names)
    if len(names) != len(set(names)):
        raise ValueError("event feature names must be unique")
    required = {
        "start_seconds",
        "duration_seconds",
        "center_offset_hz",
        "occupied_bandwidth_hz",
        "mean_cfar_db",
    }
    missing = required - set(names)
    if missing:
        raise ValueError(f"event features lack required channels: {sorted(missing)}")
    return {name: names.index(name) for name in required}


def merge_overlapping_tf_components(
    events: np.ndarray,
    channel_names: Iterable[str],
    *,
    time_quantum_seconds: float | None = None,
) -> tuple[PhysicalBurst, ...]:
    """Merge TF components by temporal interval union.

    The representative frequency component is chosen by a fixed, label-blind
    ordering: strongest mean CA-CFAR evidence, then largest occupied bandwidth,
    longest duration, earliest start, and finally original row index.
    """

    value = np.asarray(events, dtype=np.float64)
    names = tuple(str(name) for name in channel_names)
    if value.ndim != 2 or value.shape[1] != len(names):
        raise ValueError("event matrix and channel names are inconsistent")
    if not np.isfinite(value).all():
        raise ValueError("event matrix contains non-finite values")
    if len(value) == 0:
        return ()
    index = _feature_index(names)
    start = value[:, index["start_seconds"]]
    duration = value[:, index["duration_seconds"]]
    stop = start + duration
    if np.any(start < 0.0) or np.any(duration <= 0.0):
        raise ValueError("events require non-negative starts and positive durations")
    if time_quantum_seconds is not None:
        if not np.isfinite(time_quantum_seconds) or time_quantum_seconds <= 0.0:
            raise ValueError("time quantum must be finite and positive")
        start_order = np.rint(start / time_quantum_seconds).astype(np.int64)
        duration_order = np.maximum(
            1, np.rint(duration / time_quantum_seconds).astype(np.int64)
        )
        stop_order = start_order + duration_order
    else:
        start_order = start
        stop_order = stop
    order = np.lexsort((np.arange(len(value)), start_order))
    groups: list[list[int]] = []
    current: list[int] = []
    running_stop = -np.inf
    for raw_index in order:
        row = int(raw_index)
        # Touching half-open intervals are different bursts.  Detector frames
        # are exact non-overlapping sample intervals, so only strict overlap
        # establishes simultaneity.
        if current and float(start_order[row]) >= running_stop:
            groups.append(current)
            current = []
            running_stop = -np.inf
        current.append(row)
        running_stop = max(running_stop, float(stop_order[row]))
    if current:
        groups.append(current)

    quality = value[:, index["mean_cfar_db"]]
    bandwidth = value[:, index["occupied_bandwidth_hz"]]
    result: list[PhysicalBurst] = []
    for group in groups:
        representative = min(
            group,
            key=lambda row: (
                -float(quality[row]),
                -float(bandwidth[row]),
                -float(duration[row]),
                float(start[row]),
                int(row),
            ),
        )
        if time_quantum_seconds is None:
            merged_start = float(np.min(start[group]))
            merged_stop = float(np.max(stop[group]))
        else:
            merged_start = float(np.min(start_order[group]) * time_quantum_seconds)
            merged_stop = float(np.max(stop_order[group]) * time_quantum_seconds)
        result.append(
            PhysicalBurst(
                start_seconds=merged_start,
                stop_seconds=merged_stop,
                representative_index=int(representative),
                component_count=len(group),
            )
        )
    if any(
        left.stop_seconds > right.start_seconds
        for left, right in zip(result[:-1], result[1:])
    ):
        raise AssertionError("merged physical bursts still overlap")
    return tuple(result)


def centered_window_bounds(
    timestamp_seconds: float,
    *,
    window_samples: int,
    sample_rate_hz: float,
    recording_samples: int,
) -> tuple[int, int]:
    """Map an exact timestamp to a fixed-length, in-record raw-IQ window."""

    if not np.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
        raise ValueError("timestamp must be finite and non-negative")
    if window_samples < 2 or window_samples > recording_samples:
        raise ValueError("invalid raw-IQ window length")
    if sample_rate_hz <= 0.0 or recording_samples < 1:
        raise ValueError("invalid sampling metadata")
    center = int(round(float(timestamp_seconds) * float(sample_rate_hz)))
    first = center - window_samples // 2
    first = min(max(first, 0), recording_samples - window_samples)
    return first, first + window_samples


def band_adjacent_energy(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    center_offset_hz: float,
    occupied_bandwidth_hz: float,
) -> tuple[float, float, float, int, int]:
    """Return signal-band, adjacent-band, and excess energy in dB/bin.

    The adjacent mask uses equally wide guard-separated side bands whenever
    possible.  At a Nyquist edge it falls back to the available out-of-band
    bins.  Mean power per bin, rather than total power, prevents bandwidth
    itself from becoming the energy feature.
    """

    value = np.asarray(signal, dtype=np.complex64).reshape(-1)
    if len(value) not in RAW_WINDOW_SIZES:
        raise ValueError(f"raw window must have one of {RAW_WINDOW_SIZES} samples")
    if not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
        raise ValueError("raw IQ contains non-finite values")
    if sample_rate_hz <= 0.0 or occupied_bandwidth_hz <= 0.0:
        raise ValueError("sample rate and bandwidth must be positive")
    nyquist = 0.5 * float(sample_rate_hz)
    if not -nyquist <= float(center_offset_hz) <= nyquist:
        raise ValueError("event center lies outside the sampled passband")

    centered = value - np.mean(value)
    taper = np.hanning(len(centered)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(centered * taper))
    power = np.maximum(np.abs(spectrum) ** 2, np.finfo(np.float64).tiny)
    frequency = np.fft.fftshift(
        np.fft.fftfreq(len(centered), d=1.0 / float(sample_rate_hz))
    )
    bin_width = float(sample_rate_hz) / len(centered)
    half_width = max(0.5 * float(occupied_bandwidth_hz), 2.0 * bin_width)
    half_width = min(half_width, 0.45 * float(sample_rate_hz))
    distance = np.abs(frequency - float(center_offset_hz))
    signal_mask = distance <= half_width
    guard = max(bin_width, 0.05 * half_width)
    adjacent_mask = (distance > half_width + guard) & (
        distance <= 3.0 * half_width + guard
    )
    # Keep a roughly matched number of nearest adjacent bins.  This avoids a
    # broad out-of-band average when the detected signal itself is narrow.
    signal_count = int(np.sum(signal_mask))
    adjacent_rows = np.flatnonzero(adjacent_mask)
    if len(adjacent_rows) > signal_count:
        rank = np.argsort(distance[adjacent_rows], kind="stable")[:signal_count]
        chosen = adjacent_rows[rank]
        adjacent_mask = np.zeros_like(signal_mask)
        adjacent_mask[chosen] = True
    if signal_count < 2:
        nearest = np.argsort(distance, kind="stable")[:2]
        signal_mask = np.zeros_like(signal_mask)
        signal_mask[nearest] = True
        signal_count = 2
    if int(np.sum(adjacent_mask)) < 2:
        candidates = np.flatnonzero(~signal_mask)
        if len(candidates) < 2:
            raise ValueError("raw window has no usable adjacent-band bins")
        nearest = candidates[
            np.argsort(distance[candidates], kind="stable")[: max(2, signal_count)]
        ]
        adjacent_mask = np.zeros_like(signal_mask)
        adjacent_mask[nearest] = True

    signal_mean = float(np.mean(power[signal_mask]))
    adjacent_mean = float(np.mean(power[adjacent_mask]))
    signal_db = 10.0 * np.log10(max(signal_mean, np.finfo(float).tiny))
    adjacent_db = 10.0 * np.log10(max(adjacent_mean, np.finfo(float).tiny))
    return (
        float(signal_db),
        float(adjacent_db),
        float(signal_db - adjacent_db),
        int(np.sum(signal_mask)),
        int(np.sum(adjacent_mask)),
    )


def recording_local_residualize(
    energy: np.ndarray,
    center_offset_hz: np.ndarray,
    occupied_bandwidth_hz: np.ndarray,
    *,
    sample_rate_hz: float,
    center_bins: int = 16,
    bandwidth_bins: int = 4,
) -> np.ndarray:
    """Remove record-local gain, coarse hop, and bandwidth offsets.

    Median additive effects are removed in two fixed passes (center-frequency,
    then log-bandwidth).  A robust record-local scale produces unitless series.
    No label, identity, condition, or cross-record population statistic enters.
    """

    value = np.asarray(energy, dtype=np.float64)
    center = np.asarray(center_offset_hz, dtype=np.float64).reshape(-1)
    bandwidth = np.asarray(occupied_bandwidth_hz, dtype=np.float64).reshape(-1)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or center.shape != (len(value),) or bandwidth.shape != center.shape:
        raise ValueError("energy and RF-nuisance vectors are not aligned")
    if len(value) < 3 or not (
        np.isfinite(value).all()
        and np.isfinite(center).all()
        and np.isfinite(bandwidth).all()
    ):
        raise ValueError("record-local residualization requires finite observations")
    if sample_rate_hz <= 0.0 or np.any(bandwidth <= 0.0):
        raise ValueError("invalid RF nuisance metadata")

    result = value - np.median(value, axis=0, keepdims=True)
    normalized_center = np.clip(center / float(sample_rate_hz) + 0.5, 0.0, 1.0)
    center_group = np.minimum(
        center_bins - 1, np.floor(normalized_center * center_bins).astype(np.int64)
    )
    for group in range(center_bins):
        selected = center_group == group
        if int(np.sum(selected)) >= 3:
            result[selected] -= np.median(result[selected], axis=0, keepdims=True)

    log_bandwidth = np.log(np.maximum(bandwidth, SCALE_FLOOR))
    edges = np.quantile(log_bandwidth, np.linspace(0.0, 1.0, bandwidth_bins + 1))
    if edges[-1] > edges[0]:
        bandwidth_group = np.searchsorted(edges[1:-1], log_bandwidth, side="right")
        for group in range(bandwidth_bins):
            selected = bandwidth_group == group
            if int(np.sum(selected)) >= 3:
                result[selected] -= np.median(result[selected], axis=0, keepdims=True)

    result -= np.median(result, axis=0, keepdims=True)
    mad = np.median(np.abs(result), axis=0)
    standard = np.std(result, axis=0)
    scale = np.where(mad > SCALE_FLOOR, 1.4826 * mad, standard)
    scale = np.where(scale > SCALE_FLOOR, scale, 1.0)
    result /= scale[None]
    if not np.isfinite(result).all():
        raise RuntimeError("record-local residualization produced non-finite values")
    return result.astype(np.float32)


def gap_nyquist_proxy_hz(
    timestamp_seconds: np.ndarray,
    *,
    gap_quantile: float,
) -> float:
    """Return ``0.5 / quantile(positive inter-burst gap)``.

    Irregular samples have no single classical Nyquist limit.  The value is
    therefore explicitly a gap proxy, not an observability guarantee.
    """

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    if len(timestamp) < 3 or not np.isfinite(timestamp).all():
        return 0.0
    if not 0.0 < gap_quantile < 1.0:
        raise ValueError("gap quantile must lie strictly inside (0,1)")
    difference = np.diff(timestamp)
    difference = difference[difference > 0.0]
    if len(difference) < 2:
        return 0.0
    return float(0.5 / np.quantile(difference, gap_quantile))


def median_gap_nyquist_proxy_hz(timestamp_seconds: np.ndarray) -> float:
    return gap_nyquist_proxy_hz(timestamp_seconds, gap_quantile=0.5)


def q90_gap_nyquist_proxy_hz(timestamp_seconds: np.ndarray) -> float:
    """Q90-gap sampling proxy used to truncate the B1 frequency grid."""

    return gap_nyquist_proxy_hz(timestamp_seconds, gap_quantile=0.9)


def interval_support_fraction(
    timestamp_seconds: np.ndarray,
    *,
    frequency_hz: float,
) -> float:
    """Fraction of positive gaps supporting a two-samples-per-cycle rule."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    if frequency_hz <= 0.0 or len(timestamp) < 3 or not np.isfinite(timestamp).all():
        return 0.0
    difference = np.diff(timestamp)
    difference = difference[difference > 0.0]
    if len(difference) < 2:
        return 0.0
    return float(np.mean(difference <= 0.5 / float(frequency_hz)))


def generalized_lomb_power(
    timestamp_seconds: np.ndarray,
    signal: np.ndarray,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Generalized Lomb/least-squares sinusoidal explained-power spectrum."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    value = np.asarray(signal, dtype=np.float64).reshape(-1)
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    if timestamp.shape != value.shape or len(timestamp) < 3:
        raise ValueError("timestamps and slow-time signal are not aligned")
    if not (
        np.isfinite(timestamp).all()
        and np.isfinite(value).all()
        and np.isfinite(frequency).all()
    ):
        raise ValueError("Lomb inputs contain non-finite values")
    if np.any(frequency <= 0.0):
        raise ValueError("Lomb frequencies must be positive")
    centered_time = timestamp - timestamp[0]
    centered_value = value - np.mean(value)
    total = float(np.dot(centered_value, centered_value))
    if total <= SCALE_FLOOR:
        return np.zeros_like(frequency)

    phase = 2.0 * np.pi * frequency[:, None] * centered_time[None]
    cosine = np.cos(phase)
    sine = np.sin(phase)
    cosine -= np.mean(cosine, axis=1, keepdims=True)
    sine -= np.mean(sine, axis=1, keepdims=True)
    cc = np.einsum("ij,ij->i", cosine, cosine)
    ss = np.einsum("ij,ij->i", sine, sine)
    cs = np.einsum("ij,ij->i", cosine, sine)
    yc = cosine @ centered_value
    ys = sine @ centered_value
    determinant = cc * ss - cs * cs
    explained = np.zeros_like(frequency)
    valid = determinant > SCALE_FLOOR
    explained[valid] = (
        ss[valid] * yc[valid] ** 2
        + cc[valid] * ys[valid] ** 2
        - 2.0 * cs[valid] * yc[valid] * ys[valid]
    ) / determinant[valid]
    return np.clip(explained / total, 0.0, 1.0)


def _signed_time_correlation(value: np.ndarray, timestamp: np.ndarray) -> float:
    y = np.asarray(value, dtype=np.float64).reshape(-1)
    t = np.asarray(timestamp, dtype=np.float64).reshape(-1)
    y = y - np.mean(y)
    t = t - np.mean(t)
    denominator = float(np.linalg.norm(y) * np.linalg.norm(t))
    return 0.0 if denominator <= SCALE_FLOOR else float(np.dot(y, t) / denominator)


def _temporal_bin_means(
    timestamp: np.ndarray,
    value: np.ndarray,
    *,
    bins: int,
    duration_seconds: float,
) -> np.ndarray:
    edges = np.linspace(0.0, duration_seconds, bins + 1)
    group = np.minimum(bins - 1, np.searchsorted(edges[1:-1], timestamp, side="right"))
    result = np.zeros(bins, dtype=np.float64)
    populated = np.zeros(bins, dtype=bool)
    for index in range(bins):
        selected = group == index
        if np.any(selected):
            result[index] = float(np.median(value[selected]))
            populated[index] = True
    if not np.all(populated):
        known = np.flatnonzero(populated)
        if len(known):
            result[~populated] = np.interp(
                np.flatnonzero(~populated), known, result[known]
            )
    return result


def low_frequency_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for channel in ENERGY_CHANNEL_NAMES:
        for low, high in LOW_FREQUENCY_BANDS_HZ:
            token = f"{low:g}_{high:g}hz"
            names.extend((f"{channel}__{token}__mean_power", f"{channel}__{token}__max_power"))
        names.extend(
            (
                f"{channel}__peak_frequency_fraction",
                f"{channel}__peak_power",
                f"{channel}__signed_time_correlation",
                f"{channel}__late_minus_early",
                f"{channel}__temporal_bin_range",
                f"{channel}__temporal_bin_total_variation",
            )
        )
    return tuple(names)


def low_frequency_features(
    timestamp_seconds: np.ndarray,
    residual_energy: np.ndarray,
    *,
    duration_seconds: float,
) -> np.ndarray:
    """B0: 0.7--10 Hz displacement/transient features."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    energy = np.asarray(residual_energy, dtype=np.float64)
    if energy.shape != (len(timestamp), len(ENERGY_CHANNEL_NAMES)):
        raise ValueError("slow-time energy has the wrong shape")
    frequency = np.linspace(0.7, 10.0, 54, dtype=np.float64)
    result: list[float] = []
    order = np.argsort(timestamp, kind="stable")
    third = max(1, len(timestamp) // 3)
    for channel in range(energy.shape[1]):
        value = energy[:, channel]
        power = generalized_lomb_power(timestamp, value, frequency)
        for low, high in LOW_FREQUENCY_BANDS_HZ:
            selected = (frequency >= low) & (frequency <= high)
            result.extend((float(np.mean(power[selected])), float(np.max(power[selected]))))
        peak = int(np.argmax(power))
        binned = _temporal_bin_means(
            timestamp, value, bins=12, duration_seconds=duration_seconds
        )
        result.extend(
            (
                float(frequency[peak] / 10.0),
                float(power[peak]),
                _signed_time_correlation(value, timestamp),
                float(np.median(value[order[-third:]]) - np.median(value[order[:third]])),
                float(np.max(binned) - np.min(binned)),
                float(np.mean(np.abs(np.diff(binned)))),
            )
        )
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (len(low_frequency_feature_names()),) or not np.isfinite(output).all():
        raise RuntimeError("low-frequency feature contract drift")
    return output


def vibration_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for low, high in VIBRATION_BANDS_HZ:
        token = f"{low:g}_{high:g}hz"
        names.append(f"coverage__{token}")
    for low, high in VIBRATION_BANDS_HZ:
        token = f"{low:g}_{high:g}hz"
        names.append(f"interval_support_at_upper_edge__{token}")
    names.extend(("q90_gap_nyquist_fraction_220hz", "log_burst_count"))
    for channel in ENERGY_CHANNEL_NAMES:
        for low, high in VIBRATION_BANDS_HZ:
            token = f"{low:g}_{high:g}hz"
            names.extend(
                (
                    f"{channel}__{token}__mean_power",
                    f"{channel}__{token}__max_power",
                    f"{channel}__{token}__peak_fraction",
                )
            )
        names.extend((f"{channel}__vibration_peak_fraction", f"{channel}__vibration_peak_power"))
    return tuple(names)


def vibration_features(
    timestamp_seconds: np.ndarray,
    residual_energy: np.ndarray,
) -> np.ndarray:
    """B1: irregular-sampling 10--220 Hz vibration features."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    energy = np.asarray(residual_energy, dtype=np.float64)
    if energy.shape != (len(timestamp), len(ENERGY_CHANNEL_NAMES)):
        raise ValueError("slow-time energy has the wrong shape")
    nyquist = q90_gap_nyquist_proxy_hz(timestamp)
    maximum = min(220.0, nyquist)
    frequency = np.arange(10.0, maximum + 0.5001, 0.5, dtype=np.float64)
    if len(frequency) == 0:
        frequency = np.asarray((10.0,), dtype=np.float64)
    coverage = [
        float(np.clip((min(high, maximum) - low) / (high - low), 0.0, 1.0))
        for low, high in VIBRATION_BANDS_HZ
    ]
    result: list[float] = list(coverage)
    result.extend(
        interval_support_fraction(timestamp, frequency_hz=high)
        for _, high in VIBRATION_BANDS_HZ
    )
    result.extend((float(min(nyquist, 220.0) / 220.0), float(np.log1p(len(timestamp)))))
    for channel in range(energy.shape[1]):
        power = generalized_lomb_power(timestamp, energy[:, channel], frequency)
        for low, high in VIBRATION_BANDS_HZ:
            selected = (frequency >= low) & (frequency <= min(high, maximum))
            if not np.any(selected):
                result.extend((0.0, 0.0, 0.0))
                continue
            local_frequency = frequency[selected]
            local_power = power[selected]
            peak = int(np.argmax(local_power))
            result.extend(
                (
                    float(np.mean(local_power)),
                    float(np.max(local_power)),
                    float((local_frequency[peak] - low) / (high - low)),
                )
            )
        peak = int(np.argmax(power))
        result.extend((float(frequency[peak] / 220.0), float(power[peak])))
    output = np.asarray(result, dtype=np.float32)
    if output.shape != (len(vibration_feature_names()),) or not np.isfinite(output).all():
        raise RuntimeError("vibration feature contract drift")
    return output


def rate_only_feature_names() -> tuple[str, ...]:
    return (
        "log_burst_count",
        "log_burst_rate_hz",
        "iat_median_seconds",
        "iat_q90_seconds",
        "iat_q99_seconds",
        "iat_coefficient_of_variation",
        "maximum_silence_seconds",
        *(f"time_bin_fraction_{index:02d}" for index in range(16)),
    )


def rate_only_features(
    timestamp_seconds: np.ndarray,
    *,
    duration_seconds: float,
) -> np.ndarray:
    """Timestamp-only control; no energy, RF center, bandwidth, or label."""

    timestamp = np.asarray(timestamp_seconds, dtype=np.float64).reshape(-1)
    if len(timestamp) < 3 or duration_seconds <= 0.0:
        raise ValueError("rate control requires at least three timestamps")
    difference = np.diff(np.sort(timestamp))
    positive = difference[difference > 0.0]
    if len(positive) < 2:
        raise ValueError("rate control requires distinct timestamps")
    count, _ = np.histogram(timestamp, bins=np.linspace(0.0, duration_seconds, 17))
    fraction = count.astype(np.float64) / max(float(np.sum(count)), 1.0)
    result = np.concatenate(
        (
            np.asarray(
                (
                    np.log1p(len(timestamp)),
                    np.log1p(len(timestamp) / duration_seconds),
                    np.median(positive),
                    np.quantile(positive, 0.90),
                    np.quantile(positive, 0.99),
                    np.std(positive) / max(np.mean(positive), SCALE_FLOOR),
                    np.max(positive),
                )
            ),
            fraction,
        )
    ).astype(np.float32)
    if result.shape != (len(rate_only_feature_names()),) or not np.isfinite(result).all():
        raise RuntimeError("rate-only feature contract drift")
    return result


LOW_FEATURE_NAMES = low_frequency_feature_names()


VIBRATION_FEATURE_NAMES = vibration_feature_names()


RATE_FEATURE_NAMES = rate_only_feature_names()


COMBINED_FEATURE_NAMES = tuple(
    [*(f"b0__{name}" for name in LOW_FEATURE_NAMES), *(f"b1__{name}" for name in VIBRATION_FEATURE_NAMES)]
)
