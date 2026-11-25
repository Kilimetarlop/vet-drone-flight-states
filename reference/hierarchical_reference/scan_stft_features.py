"""Scan and STFT recording features (3424 + 1524 dimensions) from the micro-IQ cache.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy.fft import dct
import torch


CLASS_NAMES = (
    "LINEAR_CRUISE",
    "CURVED_TURN",
    "HOVER",
    "TAKEOFF_ASCEND",
    "DESCEND_LAND",
)


EXPECTED_COUNTS = (48, 48, 24, 24, 24)


EXPECTED_EVENTS = 168


EXPECTED_GROUPS = 8


FEATURE_VERSION = "behavior5_uniform_probe_stft_summary_v2"


NFFT = 512


HOP = 512


FRAMES = 32


BINS = 96


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _validate_cache(cache: Any) -> tuple[np.ndarray, ...]:
    required = {
        "event",
        "identity",
        "behavior",
        "class_index",
        "selected_candidate_indices",
        "scan_scores",
    }
    missing = sorted(required - set(cache.files))
    if missing:
        raise RuntimeError(f"behavior cache missing keys: {missing}")
    event = cache["event"].astype(str)
    identity = cache["identity"].astype(str)
    behavior = cache["behavior"].astype(str)
    target = np.asarray(cache["class_index"], dtype=np.int64)
    selected = np.asarray(cache["selected_candidate_indices"], dtype=np.int64)
    scan = np.asarray(cache["scan_scores"], dtype=np.float32)
    if "uniform_iq_real_imag" in cache.files:
        iq = cache["uniform_iq_real_imag"]
        if iq.shape != (EXPECTED_EVENTS, 64, 16_384, 2):
            raise RuntimeError(f"unexpected uniform-IQ shape: {iq.shape}")
        local_view = "uniform64"
    elif "iq_real_imag" in cache.files:
        iq = cache["iq_real_imag"]
        if iq.shape != (EXPECTED_EVENTS, 29, 16_384, 2):
            raise RuntimeError(f"unexpected active-IQ shape: {iq.shape}")
        local_view = "active29_fallback"
    else:
        raise RuntimeError("behavior cache has neither uniform nor active micro-IQ")
    probe = (
        np.asarray(cache["probe_features"], dtype=np.float32)
        if "probe_features" in cache.files
        else None
    )
    if event.shape != (EXPECTED_EVENTS,):
        raise RuntimeError(f"expected {EXPECTED_EVENTS} events, got {event.shape}")
    if len(set(event.tolist())) != EXPECTED_EVENTS:
        raise RuntimeError("event names are not unique")
    if len(set(identity.tolist())) != EXPECTED_GROUPS:
        raise RuntimeError("expected eight UAV/model groups")
    if selected.shape != (EXPECTED_EVENTS, 29):
        raise RuntimeError(f"unexpected selected-index shape: {selected.shape}")
    if scan.shape != (EXPECTED_EVENTS, 256) or not np.isfinite(scan).all():
        raise RuntimeError(f"invalid scan-score matrix: {scan.shape}")
    if probe is not None:
        if probe.ndim != 3 or probe.shape[:2] != (EXPECTED_EVENTS, 256):
            raise RuntimeError(f"unexpected uniform probe features: {probe.shape}")
        if not np.isfinite(probe).all():
            raise RuntimeError("uniform probe features contain non-finite values")
    if not np.array_equal(np.unique(target), np.arange(len(CLASS_NAMES))):
        raise RuntimeError(f"class indices must be 0..4, got {np.unique(target)}")
    observed = tuple(int(np.sum(target == row)) for row in range(len(CLASS_NAMES)))
    if observed != EXPECTED_COUNTS:
        raise RuntimeError(f"class-count drift: {observed}")
    expected_behavior = np.asarray([CLASS_NAMES[row] for row in target])
    if not np.array_equal(behavior, expected_behavior):
        raise RuntimeError("behavior names and class indices disagree")
    group_counts = Counter(identity.tolist())
    if set(group_counts.values()) != {21}:
        raise RuntimeError(f"each held group must have 21 events: {group_counts}")
    return event, identity, behavior, target, selected, scan, iq, probe, local_view


def build_scan_features(
    scan_scores: np.ndarray,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Create gain-robust and absolute slow-time activity descriptors."""
    score = np.asarray(scan_scores, dtype=np.float64)
    median = np.median(score, axis=1, keepdims=True)
    q25 = np.quantile(score, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(score, 0.75, axis=1, keepdims=True)
    robust = (score - median) / np.maximum(q75 - q25, 1.0e-6)
    difference = np.diff(robust, axis=1)
    frequency = dct(robust, type=2, axis=1, norm="ortho")[:, :64]

    quantiles = np.quantile(
        score, (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0),
        axis=1,
    ).T
    moments = np.stack(
        (
            score.mean(axis=1),
            score.std(axis=1),
            np.mean(np.abs(difference), axis=1),
            difference.std(axis=1),
        ),
        axis=1,
    )
    correlations = []
    for lag in (1, 2, 4, 8, 16, 32):
        left = robust[:, :-lag]
        right = robust[:, lag:]
        numerator = np.mean(left * right, axis=1)
        denominator = np.sqrt(
            np.mean(np.square(left), axis=1)
            * np.mean(np.square(right), axis=1)
        )
        correlations.append(numerator / np.maximum(denominator, 1.0e-6))
    correlation = np.stack(correlations, axis=1)

    position = np.sort(selected_indices, axis=1).astype(np.float64) / 255.0
    gaps = np.diff(
        np.concatenate(
            (np.zeros((len(position), 1)), position, np.ones((len(position), 1))),
            axis=1,
        ),
        axis=1,
    )
    selected_score = np.take_along_axis(score, selected_indices, axis=1)
    result = np.concatenate(
        (
            score,
            robust,
            difference,
            frequency,
            quantiles,
            moments,
            correlation,
            position,
            gaps,
            selected_score,
        ),
        axis=1,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite scan features")
    return result.astype(np.float32)


def build_probe_sequence_features(probe_features: np.ndarray) -> np.ndarray:
    """Summarize uniformly sampled multi-spectral probes over the full event."""
    value = np.asarray(probe_features, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != 256:
        raise RuntimeError(f"unexpected probe sequence: {value.shape}")
    median = np.median(value, axis=1, keepdims=True)
    q25 = np.quantile(value, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(value, 0.75, axis=1, keepdims=True)
    robust = (value - median) / np.maximum(q75 - q25, 1.0e-6)
    difference = np.diff(robust, axis=1)

    # 32 bins span the complete 1.4 s event at about 44 ms per pooled token.
    coarse_raw = value.reshape(len(value), 32, 8, value.shape[2]).mean(axis=2)
    coarse_robust = robust.reshape(
        len(value), 32, 8, value.shape[2]
    ).mean(axis=2)
    temporal_frequency = dct(robust, type=2, axis=1, norm="ortho")[:, :32]
    quantiles = np.quantile(
        value, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0), axis=1,
    ).transpose(1, 0, 2)
    statistics = np.stack(
        (
            value.mean(axis=1),
            value.std(axis=1),
            np.abs(difference).mean(axis=1),
            difference.std(axis=1),
        ),
        axis=1,
    )
    correlations = []
    for lag in (1, 2, 4, 8, 16, 32):
        left = robust[:, :-lag]
        right = robust[:, lag:]
        numerator = np.mean(left * right, axis=1)
        denominator = np.sqrt(
            np.mean(np.square(left), axis=1)
            * np.mean(np.square(right), axis=1)
        )
        correlations.append(numerator / np.maximum(denominator, 1.0e-6))
    correlation = np.stack(correlations, axis=1)
    result = np.concatenate(
        (
            coarse_raw.reshape(len(value), -1),
            coarse_robust.reshape(len(value), -1),
            temporal_frequency.reshape(len(value), -1),
            quantiles.reshape(len(value), -1),
            statistics.reshape(len(value), -1),
            correlation.reshape(len(value), -1),
        ),
        axis=1,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite uniform probe features")
    return result.astype(np.float32)


def _standardize_stft(value: torch.Tensor) -> torch.Tensor:
    center = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(
        dim=(-2, -1), keepdim=True, unbiased=False,
    ).clamp_min(1.0e-5)
    return (value - center) / scale


@torch.inference_mode()
def build_stft_view(iq: np.ndarray, device: torch.device) -> np.ndarray:
    """Build the local four-channel STFT view for 29 or 64 ordered windows."""
    if iq.ndim != 4 or iq.shape[2:] != (16_384, 2):
        raise RuntimeError(f"unexpected micro-IQ shape: {iq.shape}")
    events, windows = iq.shape[:2]
    output = np.empty(
        (events, windows, 4, FRAMES, BINS), dtype=np.float16,
    )
    taper = torch.hann_window(NFFT, device=device)
    batch_events = 2 if windows >= 64 else 4
    for start in range(0, events, batch_events):
        stop = min(start + batch_events, events)
        raw = torch.as_tensor(iq[start:stop], device=device).float()
        raw -= raw.mean(dim=2, keepdim=True)
        raw /= torch.sqrt(
            raw.square().mean(dim=(2, 3), keepdim=True)
        ).clamp_min(1.0e-6)
        flat = raw.reshape(-1, raw.shape[2], 2)
        signal = torch.complex(flat[..., 0], flat[..., 1])
        spectrum = torch.stft(
            signal,
            n_fft=NFFT,
            hop_length=HOP,
            win_length=NFFT,
            window=taper,
            center=False,
            onesided=False,
            return_complex=True,
        )
        spectrum = torch.fft.fftshift(spectrum, dim=1)
        spectrum = spectrum[
            :, NFFT // 2 - BINS // 2 : NFFT // 2 + BINS // 2
        ]
        log_magnitude = torch.log1p(spectrum.abs())
        delta = torch.zeros_like(log_magnitude)
        delta[:, :, 1:] = log_magnitude[:, :, 1:] - log_magnitude[:, :, :-1]
        phase = torch.angle(spectrum[:, :, 1:] * spectrum[:, :, :-1].conj())
        phase_sin = torch.zeros_like(log_magnitude)
        phase_cos = torch.ones_like(log_magnitude)
        phase_sin[:, :, 1:] = torch.sin(phase)
        phase_cos[:, :, 1:] = torch.cos(phase)
        feature = torch.stack(
            (
                _standardize_stft(log_magnitude),
                _standardize_stft(delta),
                phase_sin,
                phase_cos,
            ),
            dim=1,
        ).permute(0, 1, 3, 2)
        output[start:stop] = feature.reshape(
            stop - start, windows, 4, FRAMES, BINS,
        ).half().cpu().numpy()
    return output


def build_stft_features(view: np.ndarray) -> np.ndarray:
    """Pool the four-channel STFT view into recording-level descriptors."""
    if view.ndim != 5 or view.shape[2:] != (4, FRAMES, BINS):
        raise RuntimeError(f"unexpected STFT view: {view.shape}")
    rows: list[np.ndarray] = []
    for event_view in view:
        value = np.asarray(event_view, dtype=np.float32)
        log_magnitude = value[:, 0]
        delta = value[:, 1]
        phase_sin = value[:, 2]
        phase_cos = value[:, 3]

        frequency_summary = np.concatenate(
            (
                log_magnitude.mean(axis=(0, 1)),
                log_magnitude.std(axis=(0, 1)),
                np.abs(delta).mean(axis=(0, 1)),
                np.sqrt(
                    np.square(phase_sin.mean(axis=(0, 1)))
                    + np.square(phase_cos.mean(axis=(0, 1)))
                ),
            )
        )

        # Twelve coarse spectral bands retain the chronological order of all
        # 29 selected windows without making the classifier excessively wide.
        windows = len(log_magnitude)
        band = log_magnitude.mean(axis=1).reshape(windows, 12, 8).mean(axis=2)
        token = np.concatenate(
            (
                band,
                log_magnitude.std(axis=(1, 2))[:, None],
                np.abs(delta).mean(axis=(1, 2))[:, None],
                np.sqrt(
                    np.square(phase_sin.mean(axis=(1, 2)))
                    + np.square(phase_cos.mean(axis=(1, 2)))
                )[:, None],
            ),
            axis=1,
        )
        temporal_dct = dct(token, type=2, axis=0, norm="ortho")[:8]
        token_statistics = np.concatenate(
            (
                token.mean(axis=0),
                token.std(axis=0),
                np.quantile(token, 0.10, axis=0),
                np.quantile(token, 0.90, axis=0),
            )
        )
        rows.append(
            np.concatenate(
                (
                    frequency_summary,
                    token.ravel(),
                    temporal_dct.ravel(),
                    token_statistics,
                )
            )
        )
    result = np.stack(rows).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite STFT features")
    return result


def _feature_cache_matches(path: Path, event: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (
                str(cache["feature_version"].item()) == FEATURE_VERSION
                and np.array_equal(cache["event"].astype(str), event)
                and cache["global"].shape[0] == len(event)
                and cache["stft"].shape[0] == len(event)
            )
    except (KeyError, OSError, ValueError):
        return False


def load_or_build_features(
    cache_path: Path,
    feature_path: Path,
    device: torch.device,
    force: bool,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], dict[str, Any]]:
    started = time.perf_counter()
    with np.load(cache_path, allow_pickle=False, mmap_mode="r") as cache:
        metadata = _validate_cache(cache)
        (
            event,
            identity,
            behavior,
            target,
            selected,
            scan_score,
            iq,
            probe,
            local_view,
        ) = metadata
        scan = build_scan_features(scan_score, selected)
        global_feature = (
            np.concatenate((scan, build_probe_sequence_features(probe)), axis=1)
            if probe is not None
            else scan
        )
        if force or not _feature_cache_matches(feature_path, event):
            stft_view = build_stft_view(iq, device)
            stft = build_stft_features(stft_view)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = feature_path.with_name(
                f"{feature_path.stem}.{os.getpid()}.tmp.npz"
            )
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    feature_version=np.asarray(FEATURE_VERSION),
                    event=event,
                    stft=stft,
                    **{"global": global_feature},
                )
            os.replace(temporary, feature_path)
            feature_status = "built"
        else:
            with np.load(feature_path, allow_pickle=False) as features:
                cached_global = np.asarray(features["global"], dtype=np.float32)
                if not np.allclose(
                    global_feature, cached_global, rtol=1.0e-5, atol=1.0e-6,
                ):
                    raise RuntimeError("cached global features differ from source cache")
                stft = np.asarray(features["stft"], dtype=np.float32)
            feature_status = "existing"
    feature = {
        "global": global_feature,
        "stft": stft,
        "fusion": np.concatenate((global_feature, stft), axis=1),
    }
    info = {
        "status": feature_status,
        "version": FEATURE_VERSION,
        "local_window_view": local_view,
        "uniform_probe_features": probe is not None,
        "path": str(feature_path.resolve()),
        "dimensions": {name: int(value.shape[1]) for name, value in feature.items()},
        "seconds": time.perf_counter() - started,
    }
    return feature, metadata[:6], info
