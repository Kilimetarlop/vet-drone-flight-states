"""Label-free burst detection on the subband power of one recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RFEventDetectorConfig:
    """Physical and statistical constants for the generic event detector."""

    sample_rate_hz: float = 100_000_000.0
    fft_size: int = 2_048
    hop_size: int = 2_048
    subbands: int = 64
    cut_lengths_frames: tuple[int, ...] = (1, 4, 16, 32)
    guard_frames: int = 16
    training_frames: int = 96
    minimum_training_fraction: float = 0.50
    cfar_threshold_db: float = 6.0
    noise_quantile: float = 0.20
    minimum_noise_excess_db: float = 6.0
    close_time_frames: int = 3
    close_frequency_bands: int = 1
    minimum_duration_frames: int = 2
    minimum_frequency_bands: int = 2
    minimum_component_pixels: int = 4

    def validate(self) -> None:
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if self.fft_size < 32 or self.hop_size < 1:
            raise ValueError("fft_size and hop_size must be positive")
        if self.fft_size % self.subbands:
            raise ValueError("fft_size must be divisible by subbands")
        if not self.cut_lengths_frames or any(
            int(value) < 1 for value in self.cut_lengths_frames
        ):
            raise ValueError("cut lengths must be positive")
        if self.guard_frames < 0 or self.training_frames < 1:
            raise ValueError("guard/training frames are invalid")
        if not 0.0 < self.minimum_training_fraction <= 1.0:
            raise ValueError("minimum_training_fraction must lie in (0,1]")
        if not 0.0 < self.noise_quantile < 0.5:
            raise ValueError("noise_quantile must lie in (0,0.5)")
        if not math.isfinite(self.cfar_threshold_db):
            raise ValueError("cfar_threshold_db must be finite")
        if not math.isfinite(self.minimum_noise_excess_db):
            raise ValueError("minimum_noise_excess_db must be finite")
        if self.close_time_frames < 1 or self.close_frequency_bands < 1:
            raise ValueError("closing dimensions must be positive")
        if (
            self.minimum_duration_frames < 1
            or self.minimum_frequency_bands < 1
            or self.minimum_component_pixels < 1
        ):
            raise ValueError("component minima must be positive")

    @property
    def frame_duration_seconds(self) -> float:
        return self.hop_size / self.sample_rate_hz

    @property
    def subband_width_hz(self) -> float:
        return self.sample_rate_hz / self.subbands

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RFEvent:
    """One connected time--frequency RF event."""

    start_frame: int
    stop_frame: int
    first_subband: int
    stop_subband: int
    active_pixels: int
    box_pixels: int
    mean_cfar_db: float
    peak_cfar_db: float
    mean_noise_excess_db: float
    peak_noise_excess_db: float
    center_offset_hz: float
    occupied_bandwidth_hz: float
    center_drift_hz: float
    spectral_entropy: float
    spectral_flatness: float

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame

    @property
    def frequency_bands(self) -> int:
        return self.stop_subband - self.first_subband

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class RFEventDetection:
    events: tuple[RFEvent, ...]
    active_pixel_fraction: float
    active_frame_fraction: float
    noise_floor_db: np.ndarray
    maximum_cfar_db: np.ndarray
    edge_blind_frames: int


def _validate_power(power: np.ndarray, config: RFEventDetectorConfig) -> np.ndarray:
    value = np.asarray(power, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != config.subbands:
        raise ValueError(
            f"power must have shape (frames,{config.subbands}), got {value.shape}"
        )
    if value.shape[0] < max(config.cut_lengths_frames):
        raise ValueError("power map is shorter than the largest CUT")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("power must be finite and non-negative")
    return np.maximum(value, np.finfo(np.float32).tiny)


def temporal_ca_cfar(
    power: np.ndarray,
    *,
    cut_length: int,
    guard_length: int,
    training_length: int,
    minimum_training_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CUT/reference ratios, CUT means, and reference-cell counts.

    The CUT and guard cells are excluded explicitly.  Available cells at a
    physical recording edge are used only when the requested minimum fraction
    is present.
    """

    value = np.asarray(power, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("power must be two-dimensional")
    if cut_length < 1 or cut_length > value.shape[0]:
        raise ValueError("cut_length is outside the recording")
    if guard_length < 0 or training_length < 1:
        raise ValueError("guard/training length is invalid")
    if not 0.0 < minimum_training_fraction <= 1.0:
        raise ValueError("minimum_training_fraction must lie in (0,1]")

    prefix = np.concatenate(
        (
            np.zeros((1, value.shape[1]), dtype=np.float64),
            np.cumsum(value, axis=0, dtype=np.float64),
        ),
        axis=0,
    )
    starts = np.arange(value.shape[0] - cut_length + 1, dtype=np.int64)
    stops = starts + cut_length
    cut_mean = (prefix[stops] - prefix[starts]) / cut_length

    pre_stop = np.maximum(0, starts - guard_length)
    pre_start = np.maximum(0, pre_stop - training_length)
    post_start = np.minimum(value.shape[0], stops + guard_length)
    post_stop = np.minimum(value.shape[0], post_start + training_length)
    reference_sum = (prefix[pre_stop] - prefix[pre_start]) + (
        prefix[post_stop] - prefix[post_start]
    )
    counts = (pre_stop - pre_start + post_stop - post_start).astype(np.int32)
    required = max(
        1,
        int(math.ceil(2 * training_length * minimum_training_fraction)),
    )
    valid = counts >= required
    reference = np.full_like(cut_mean, np.inf, dtype=np.float64)
    reference[valid] = reference_sum[valid] / counts[valid, None]
    ratio = cut_mean / np.maximum(reference, np.finfo(np.float64).tiny)
    ratio[~valid] = 0.0
    return (
        ratio.astype(np.float32, copy=False),
        cut_mean.astype(np.float32, copy=False),
        counts,
    )


def _expand_cut_hits(hits: np.ndarray, cut_length: int, frames: int) -> np.ndarray:
    """Expand positive CUT starts to their covered frames without Python loops."""

    value = np.asarray(hits, dtype=bool)
    expected = frames - cut_length + 1
    if value.ndim != 2 or value.shape[0] != expected:
        raise ValueError("CUT hit matrix shape is inconsistent")
    difference = np.zeros((frames + 1, value.shape[1]), dtype=np.int16)
    difference[:expected] += value.astype(np.int16)
    difference[cut_length : cut_length + expected] -= value.astype(np.int16)
    return np.cumsum(difference[:-1], axis=0) > 0


def _expand_cut_scores(
    scores: np.ndarray,
    hits: np.ndarray,
    cut_length: int,
    frames: int,
) -> np.ndarray:
    """Assign each positive CUT score to exactly its forward-covered frames."""

    value = np.asarray(scores, dtype=np.float32)
    selected = np.asarray(hits, dtype=bool)
    expected = frames - cut_length + 1
    if value.shape != selected.shape or value.shape[0] != expected:
        raise ValueError("CUT score matrix shape is inconsistent")
    result = np.ones((frames, value.shape[1]), dtype=np.float32)
    result[:expected] = np.where(selected, value, 1.0)
    if cut_length > 1:
        # scipy's positive origin shifts the footprint forward.  This origin
        # maps a score at CUT start t to [t, t + cut_length), including for
        # even window lengths.
        result = ndimage.maximum_filter1d(
            result,
            size=cut_length,
            axis=0,
            origin=(cut_length - 1) // 2,
            mode="constant",
            cval=1.0,
        )
    return result


def _weighted_spectral_summary(
    component_power: np.ndarray,
    component_mask: np.ndarray,
    *,
    first_subband: int,
    config: RFEventDetectorConfig,
) -> tuple[float, float, float, float, float]:
    weights = np.where(component_mask, component_power, 0.0).astype(np.float64)
    band_weights = weights.sum(axis=0)
    total = float(band_weights.sum())
    if total <= np.finfo(float).tiny:
        band_weights = component_mask.sum(axis=0).astype(np.float64)
        total = float(max(band_weights.sum(), 1.0))
    probability = band_weights / total
    local_indices = np.arange(len(band_weights), dtype=np.float64)
    global_centers = (
        first_subband + local_indices + 0.5 - 0.5 * config.subbands
    ) * config.subband_width_hz
    center = float(np.dot(probability, global_centers))
    cumulative = np.cumsum(probability)
    lower = int(np.searchsorted(cumulative, 0.05, side="left"))
    upper = int(np.searchsorted(cumulative, 0.95, side="left"))
    bandwidth = float((upper - lower + 1) * config.subband_width_hz)
    positive = probability > 0.0
    entropy = float(
        -np.sum(probability[positive] * np.log(probability[positive]))
        / max(math.log(len(probability)), np.finfo(float).tiny)
    ) if len(probability) > 1 else 0.0
    arithmetic = float(np.mean(np.maximum(band_weights, np.finfo(float).tiny)))
    geometric = float(np.exp(np.mean(np.log(np.maximum(band_weights, np.finfo(float).tiny)))))
    flatness = geometric / max(arithmetic, np.finfo(float).tiny)

    frame_total = weights.sum(axis=1)
    valid_frames = frame_total > 0.0
    frame_centers = np.full(len(frame_total), np.nan, dtype=np.float64)
    if np.any(valid_frames):
        frame_centers[valid_frames] = (
            weights[valid_frames] @ global_centers
        ) / frame_total[valid_frames]
    observed = frame_centers[np.isfinite(frame_centers)]
    drift = float(observed[-1] - observed[0]) if len(observed) > 1 else 0.0
    return center, bandwidth, drift, entropy, flatness


def detect_rf_events(
    power: np.ndarray,
    config: RFEventDetectorConfig | None = None,
) -> RFEventDetection:
    """Detect variable-length events from a linear-power time--frequency map."""

    settings = RFEventDetectorConfig() if config is None else config
    settings.validate()
    value = _validate_power(power, settings)
    frames, bands = value.shape
    noise_floor = np.quantile(value, settings.noise_quantile, axis=0).astype(
        np.float32
    )
    absolute_ratio = value / np.maximum(
        noise_floor[None], np.finfo(np.float32).tiny
    )
    active = np.zeros((frames, bands), dtype=bool)
    maximum_ratio = np.ones((frames, bands), dtype=np.float32)
    cfar_threshold = 10.0 ** (settings.cfar_threshold_db / 10.0)
    noise_threshold = 10.0 ** (settings.minimum_noise_excess_db / 10.0)

    for cut_length in settings.cut_lengths_frames:
        ratio, cut_mean, _ = temporal_ca_cfar(
            value,
            cut_length=cut_length,
            guard_length=settings.guard_frames,
            training_length=settings.training_frames,
            minimum_training_fraction=settings.minimum_training_fraction,
        )
        starts = ratio.shape[0]
        absolute_cut_ratio = cut_mean / np.maximum(
            noise_floor[None], np.finfo(np.float32).tiny
        )
        hits = (ratio >= cfar_threshold) & (absolute_cut_ratio >= noise_threshold)
        # A long CUT establishes contextual evidence, but it must not paint
        # the quiet samples between two packets as active.  Retain only the
        # covered frames whose instantaneous subband power also clears the
        # absolute noise-floor gate.
        active |= _expand_cut_hits(hits, cut_length, frames) & (
            absolute_ratio >= noise_threshold
        )
        expanded_ratio = _expand_cut_scores(
            ratio, hits, cut_length, frames
        )
        maximum_ratio = np.maximum(maximum_ratio, expanded_ratio)

    seed_active = active.copy()
    structure = np.ones(
        (settings.close_time_frames, settings.close_frequency_bands),
        dtype=bool,
    )
    # A false border prevents closing from inventing foreground just outside
    # the capture and bridging a near-edge event through quiet frames to frame
    # zero.  OR-ing the original seeds back in retains genuine detections that
    # actually touch a time/frequency boundary.  Four-connected labelling then
    # prevents two merely diagonal pixels from joining unrelated packets;
    # genuine wideband events remain connected through shared edges.
    grouping_active = ndimage.binary_closing(
        seed_active,
        structure=structure,
        border_value=0,
    )
    grouping_active |= seed_active
    connectivity = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.int8
    )
    labels, count = ndimage.label(grouping_active, structure=connectivity)
    objects = ndimage.find_objects(labels, max_label=count)
    tiny = np.finfo(np.float32).tiny
    cfar_db = 10.0 * np.log10(np.maximum(maximum_ratio, tiny))
    excess_db = 10.0 * np.log10(np.maximum(absolute_ratio, tiny))

    events: list[RFEvent] = []
    accepted_active = np.zeros_like(seed_active)
    for label_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        time_slice, frequency_slice = slices
        duration = int(time_slice.stop - time_slice.start)
        frequency_bands = int(frequency_slice.stop - frequency_slice.start)
        component_labels = labels[time_slice, frequency_slice]
        grouping_component = component_labels == label_index
        # Closing is used only to assign nearby seed pixels to one event.  Its
        # synthetic bridge pixels must never contribute RF power, CFAR,
        # bandwidth, entropy, or activity statistics.
        component_mask = (
            seed_active[time_slice, frequency_slice] & grouping_component
        )
        pixels = int(component_mask.sum())
        if (
            duration < settings.minimum_duration_frames
            or frequency_bands < settings.minimum_frequency_bands
            or pixels < settings.minimum_component_pixels
        ):
            continue
        accepted_view = accepted_active[time_slice, frequency_slice]
        accepted_view[component_mask] = True
        component_power = value[time_slice, frequency_slice]
        component_cfar = cfar_db[time_slice, frequency_slice][component_mask]
        component_excess = excess_db[time_slice, frequency_slice][component_mask]
        center, bandwidth, drift, entropy, flatness = _weighted_spectral_summary(
            component_power,
            component_mask,
            first_subband=int(frequency_slice.start),
            config=settings,
        )
        events.append(
            RFEvent(
                start_frame=int(time_slice.start),
                stop_frame=int(time_slice.stop),
                first_subband=int(frequency_slice.start),
                stop_subband=int(frequency_slice.stop),
                active_pixels=pixels,
                box_pixels=duration * frequency_bands,
                mean_cfar_db=float(np.mean(component_cfar)),
                peak_cfar_db=float(np.max(component_cfar)),
                mean_noise_excess_db=float(np.mean(component_excess)),
                peak_noise_excess_db=float(np.max(component_excess)),
                center_offset_hz=center,
                occupied_bandwidth_hz=bandwidth,
                center_drift_hz=drift,
                spectral_entropy=entropy,
                spectral_flatness=flatness,
            )
        )
    events.sort(key=lambda row: (row.start_frame, row.center_offset_hz))
    return RFEventDetection(
        events=tuple(events),
        active_pixel_fraction=float(np.mean(accepted_active)),
        active_frame_fraction=float(np.mean(np.any(accepted_active, axis=1))),
        noise_floor_db=(10.0 * np.log10(np.maximum(noise_floor, tiny))).astype(
            np.float32
        ),
        maximum_cfar_db=cfar_db,
        edge_blind_frames=settings.guard_frames + settings.training_frames,
    )
