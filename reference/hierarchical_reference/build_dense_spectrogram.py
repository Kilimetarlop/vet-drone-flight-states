"""Builds dense spectrogram excerpts and their scalar descriptors for every recording.
"""

from __future__ import annotations

from .dataset import DEFAULT_DATASET_ROOT as _DATASET_ROOT  # set DRFF_R2_DATASET1 to the Dataset 1 folder

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import h5py
import numpy as np
import torch

from .dataset import BEHAVIOR5_CLASSES, load_drff_r2_dataset1_behavior5_inventory


SCRIPT = Path(__file__).resolve()


WORKSPACE = SCRIPT.parents[1]


EVENTS = 168


SEGMENTS = 3


SEGMENT_SAMPLES = 10_000_000


SEGMENT_STARTS = (15_000_000, 65_000_000, 115_000_000)


NFFT = 1_024


TIME_BINS = 128


FREQUENCY_BINS = 128


FRAMES_PER_TIME_BIN = 76


USED_FRAMES = TIME_BINS * FRAMES_PER_TIME_BIN


CHANNELS = 3


SCHEMA = "drff-r2-dataset1-behavior5-dense-spectrogram/v1"


SEGMENT_FEATURE_NAMES = (
    "log_rms",
    "frame_log_power_mean",
    "frame_log_power_std",
    "frame_log_power_q90_minus_q10",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


@torch.inference_mode()
def dense_spectrogram(
    real: np.ndarray,
    imag: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one continuous 0.1 s I/Q excerpt into a compact full-band map."""
    if real.shape != (SEGMENT_SAMPLES,) or imag.shape != (SEGMENT_SAMPLES,):
        raise RuntimeError(f"unexpected dense segment: {real.shape}/{imag.shape}")
    signal = torch.complex(
        torch.as_tensor(real, device=device),
        torch.as_tensor(imag, device=device),
    )
    signal = signal - signal.mean()
    rms = torch.sqrt(signal.abs().square().mean().clamp_min(1.0e-12))
    frames = signal[: (SEGMENT_SAMPLES // NFFT) * NFFT].reshape(-1, NFFT)
    taper = torch.hann_window(NFFT, periodic=True, device=device)
    spectrum = torch.fft.fftshift(
        torch.fft.fft(frames * taper[None, :], dim=1), dim=1,
    )
    power = spectrum.abs().square().clamp_min(1.0e-12)
    del spectrum

    discard = power.shape[0] - USED_FRAMES
    first = discard // 2
    power = power[first : first + USED_FRAMES]
    pooled = power.reshape(
        TIME_BINS,
        FRAMES_PER_TIME_BIN,
        FREQUENCY_BINS,
        NFFT // FREQUENCY_BINS,
    ).mean(dim=(1, 3))
    log_power = torch.log(pooled)
    relative = log_power - log_power.mean()
    relative = relative / relative.std(unbiased=False).clamp_min(1.0e-5)
    temporal_delta = torch.zeros_like(relative)
    temporal_delta[1:] = relative[1:] - relative[:-1]
    temporal_delta /= temporal_delta.std(unbiased=False).clamp_min(1.0e-5)
    frequency_delta = torch.zeros_like(relative)
    frequency_delta[:, 1:] = relative[:, 1:] - relative[:, :-1]
    frequency_delta /= frequency_delta.std(unbiased=False).clamp_min(1.0e-5)
    image = torch.stack((relative, temporal_delta, frequency_delta))

    frame_power = frames.abs().square().mean(dim=1).clamp_min(1.0e-12)
    frame_log = torch.log(frame_power)
    mean_spectrum = power.mean(dim=0)
    probability = mean_spectrum / mean_spectrum.sum().clamp_min(1.0e-12)
    frequency = torch.linspace(-1.0, 1.0, NFFT, device=device)
    centroid = (probability * frequency).sum()
    bandwidth = torch.sqrt(
        (probability * (frequency - centroid).square()).sum().clamp_min(0.0)
    )
    entropy = -(
        probability * torch.log(probability.clamp_min(1.0e-12))
    ).sum() / np.log(NFFT)
    flatness = torch.exp(torch.log(mean_spectrum).mean()) / mean_spectrum.mean()
    feature = torch.stack(
        (
            torch.log(rms.clamp_min(1.0e-12)),
            frame_log.mean(),
            frame_log.std(unbiased=False),
            torch.quantile(frame_log, 0.90) - torch.quantile(frame_log, 0.10),
            entropy,
            centroid,
            bandwidth,
            flatness,
        )
    )
    image_np = image.half().cpu().numpy()
    feature_np = feature.float().cpu().numpy()
    if (
        image_np.shape != (CHANNELS, TIME_BINS, FREQUENCY_BINS)
        or feature_np.shape != (len(SEGMENT_FEATURE_NAMES),)
        or not np.isfinite(image_np).all()
        or not np.isfinite(feature_np).all()
    ):
        raise RuntimeError("invalid dense spectrogram output")
    return image_np, feature_np


def _metadata(rows: tuple[Any, ...]) -> dict[str, np.ndarray]:
    return {
        "event": np.asarray([row.event for row in rows]),
        "identity": np.asarray([row.identity for row in rows]),
        "behavior": np.asarray([row.behavior for row in rows]),
        "class_index": np.asarray([row.class_index for row in rows], dtype=np.int64),
        "condition": np.asarray([row.condition for row in rows], dtype=np.int64),
        "class_names": np.asarray(BEHAVIOR5_CLASSES),
        "segment_starts": np.asarray(SEGMENT_STARTS, dtype=np.int64),
        "segment_feature_names": np.asarray(SEGMENT_FEATURE_NAMES),
        "schema": np.asarray(SCHEMA),
    }


def _open_partial(
    directory: Path,
    metadata: dict[str, np.ndarray],
) -> tuple[np.memmap, np.memmap, np.memmap]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in metadata.items():
        path = directory / f"{name}.npy"
        if not path.exists():
            np.save(path, expected, allow_pickle=False)
        if not np.array_equal(np.load(path, allow_pickle=False), expected):
            raise RuntimeError(f"partial metadata mismatch: {name}")
    specifications = {
        "spectrogram": (
            np.float16,
            (EVENTS, SEGMENTS, CHANNELS, TIME_BINS, FREQUENCY_BINS),
            0.0,
        ),
        "segment_features": (
            np.float32,
            (EVENTS, SEGMENTS, len(SEGMENT_FEATURE_NAMES)),
            np.nan,
        ),
        "completed": (np.bool_, (EVENTS,), False),
    }
    output = []
    for name, (dtype, shape, fill) in specifications.items():
        path = directory / f"{name}.npy"
        if not path.exists():
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            value[:] = fill
            value.flush()
            del value
        value = np.lib.format.open_memmap(path, mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise RuntimeError(f"partial array drift: {name} {value.shape} {value.dtype}")
        output.append(value)
    return tuple(output)  # type: ignore[return-value]


def _validate_output(path: Path, metadata: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as cache:
        for name, expected in metadata.items():
            if name not in cache.files or not np.array_equal(cache[name], expected):
                raise RuntimeError(f"dense cache metadata mismatch: {name}")
        if cache["spectrogram"].shape != (
            EVENTS, SEGMENTS, CHANNELS, TIME_BINS, FREQUENCY_BINS,
        ):
            raise RuntimeError("dense cache spectrogram shape mismatch")
        if cache["segment_features"].shape != (
            EVENTS, SEGMENTS, len(SEGMENT_FEATURE_NAMES),
        ):
            raise RuntimeError("dense cache scalar-feature shape mismatch")
        if not np.isfinite(cache["spectrogram"]).all():
            raise RuntimeError("dense cache contains non-finite maps")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows = load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    metadata = _metadata(rows)
    if args.output.exists():
        _validate_output(args.output, metadata)
        return {"status": "existing", "path": str(args.output.resolve())}
    spectrogram, segment_features, completed = _open_partial(
        args.partial_dir, metadata,
    )
    resumed = int(completed.sum())
    device = _device(args.device)
    for row_index, row in enumerate(rows):
        if bool(completed[row_index]):
            continue
        with h5py.File(row.path, "r") as source:
            i_data = source["RF0_I"]
            q_data = source["RF0_Q"]
            if i_data.shape != (140_000_000, 1) or q_data.shape != i_data.shape:
                raise RuntimeError(f"unexpected recording shape: {row.event}")
            for segment_index, segment_start in enumerate(SEGMENT_STARTS):
                real = np.asarray(
                    i_data[segment_start : segment_start + SEGMENT_SAMPLES, 0],
                    dtype=np.float32,
                )
                imag = np.asarray(
                    q_data[segment_start : segment_start + SEGMENT_SAMPLES, 0],
                    dtype=np.float32,
                )
                image, features = dense_spectrogram(real, imag, device)
                spectrogram[row_index, segment_index] = image
                segment_features[row_index, segment_index] = features
        spectrogram.flush()
        segment_features.flush()
        completed[row_index] = True
        completed.flush()
        print(
            f"dense behavior {int(completed.sum())}/{EVENTS} {row.event} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            **metadata,
            spectrogram=np.asarray(spectrogram),
            segment_features=np.asarray(segment_features),
        )
    os.replace(temporary, args.output)
    _validate_output(args.output, metadata)
    return {
        "status": "complete",
        "path": str(args.output.resolve()),
        "bytes": int(args.output.stat().st_size),
        "events": EVENTS,
        "segments_per_event": SEGMENTS,
        "continuous_seconds_per_segment": SEGMENT_SAMPLES / 100_000_000,
        "resumed_events": resumed,
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
        default=WORKSPACE / "tmp/drff_r2_dataset1_behavior5_dense_v1.partial",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "tmp/drff_r2_dataset1_behavior5_dense_v1.npz",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
