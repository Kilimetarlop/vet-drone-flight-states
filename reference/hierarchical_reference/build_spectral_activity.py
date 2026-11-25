"""Builds the spectral-activity tracks of every recording.
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


SAMPLE_RATE = 100_000_000


RECORDING_SAMPLES = 140_000_000


NFFT = 2_048


FRAMES_PER_TOKEN = 48


TOKEN_SAMPLES = NFFT * FRAMES_PER_TOKEN


TOKENS = 1_424


USED_SAMPLES = TOKENS * TOKEN_SAMPLES


TRAILING_SAMPLES = RECORDING_SAMPLES - USED_SAMPLES


SUBBANDS = 32


BINS_PER_SUBBAND = NFFT // SUBBANDS


SCHEMA = "drff-r2-dataset1-behavior5-spectral-activity/v1"


TOKEN_FEATURE_NAMES = (
    *(f"relative_log_power_band_{index:02d}" for index in range(SUBBANDS)),
    "log_total_power",
    "occupied_bandwidth_90",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "peak_band_position",
    "peak_dominance",
    "occupied_band_fraction",
    "hop_rate",
    "hop_distance_mean",
    "hop_distance_q90",
    "hop_distance_max",
    "peak_unique_fraction",
    "dwell_mean_ratio",
    "dwell_max_ratio",
    "spectral_flux_mean",
    "spectral_flux_q90",
    "spectral_flux_max",
    "frame_log_power_std",
    "frame_log_power_q90_minus_q10",
    "frame_power_burst_duty",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return result


def _maximum_dwell_ratio(peak_index: np.ndarray) -> np.ndarray:
    result = np.empty(len(peak_index), dtype=np.float32)
    for row_index, row in enumerate(peak_index):
        edges = np.flatnonzero(np.r_[True, row[1:] != row[:-1], True])
        result[row_index] = float(np.diff(edges).max() / FRAMES_PER_TOKEN)
    return result


@torch.inference_mode()
def spectral_activity_tokens(
    real: np.ndarray,
    imag: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Convert one or more complete tokens to gain-robust activity features."""
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    if real.ndim != 1 or imag.shape != real.shape:
        raise ValueError(f"I/Q shape mismatch: {real.shape}/{imag.shape}")
    if len(real) == 0 or len(real) % TOKEN_SAMPLES:
        raise ValueError("I/Q batch must contain complete 98,304-sample tokens")
    token_count = len(real) // TOKEN_SAMPLES
    signal = torch.complex(
        torch.as_tensor(real, device=device), torch.as_tensor(imag, device=device),
    ).reshape(token_count, FRAMES_PER_TOKEN, NFFT)
    signal = signal - signal.mean(dim=2, keepdim=True)
    taper = torch.hann_window(NFFT, periodic=True, device=device)
    spectrum = torch.fft.fftshift(torch.fft.fft(signal * taper, dim=2), dim=2)
    power = spectrum.abs().square().clamp_min(1.0e-20)
    del spectrum, signal
    band_power = power.reshape(
        token_count, FRAMES_PER_TOKEN, SUBBANDS, BINS_PER_SUBBAND,
    ).mean(dim=3)
    token_band_power = band_power.mean(dim=1).clamp_min(1.0e-20)
    log_band = torch.log(token_band_power)
    relative_log_band = log_band - log_band.mean(dim=1, keepdim=True)

    probability = token_band_power / token_band_power.sum(dim=1, keepdim=True)
    band_position = torch.linspace(-1.0, 1.0, SUBBANDS, device=device)
    centroid = (probability * band_position).sum(dim=1)
    bandwidth = torch.sqrt(
        (probability * (band_position[None] - centroid[:, None]).square())
        .sum(dim=1)
        .clamp_min(0.0)
    )
    entropy = -(
        probability * torch.log(probability.clamp_min(1.0e-20))
    ).sum(dim=1) / np.log(SUBBANDS)
    flatness = torch.exp(torch.log(token_band_power).mean(dim=1)) / token_band_power.mean(dim=1)
    cumulative = probability.cumsum(dim=1)
    lower = (cumulative < 0.05).sum(dim=1)
    upper = (cumulative < 0.95).sum(dim=1)
    occupied_width = (upper - lower + 1).float() / SUBBANDS
    token_peak = probability.argmax(dim=1)
    peak_position = band_position[token_peak]
    peak_dominance = probability.amax(dim=1)
    occupied_fraction = (probability > (1.0 / SUBBANDS)).float().mean(dim=1)

    frame_probability = band_power / band_power.sum(dim=2, keepdim=True).clamp_min(1.0e-20)
    frame_peak = frame_probability.argmax(dim=2)
    hop = torch.abs(torch.diff(frame_peak, dim=1)).float()
    hop_rate = (hop > 0).float().mean(dim=1)
    hop_mean = hop.mean(dim=1) / (SUBBANDS - 1)
    hop_q90 = torch.quantile(hop, 0.90, dim=1) / (SUBBANDS - 1)
    hop_max = hop.amax(dim=1) / (SUBBANDS - 1)
    sorted_peak = torch.sort(frame_peak, dim=1).values
    unique_count = 1 + (torch.diff(sorted_peak, dim=1) != 0).sum(dim=1)
    unique_fraction = unique_count.float() / SUBBANDS
    run_count = 1 + (torch.diff(frame_peak, dim=1) != 0).sum(dim=1)
    dwell_mean = 1.0 / run_count.float()
    dwell_max = torch.as_tensor(
        _maximum_dwell_ratio(frame_peak.detach().cpu().numpy()), device=device,
    )

    flux = 0.5 * torch.abs(torch.diff(frame_probability, dim=1)).sum(dim=2)
    flux_mean = flux.mean(dim=1)
    flux_q90 = torch.quantile(flux, 0.90, dim=1)
    flux_max = flux.amax(dim=1)
    frame_power = band_power.sum(dim=2).clamp_min(1.0e-20)
    frame_log_power = torch.log(frame_power)
    frame_log_std = frame_log_power.std(dim=1, unbiased=False)
    frame_log_range = (
        torch.quantile(frame_log_power, 0.90, dim=1)
        - torch.quantile(frame_log_power, 0.10, dim=1)
    )
    frame_median = frame_log_power.median(dim=1, keepdim=True).values
    frame_mad = (
        torch.abs(frame_log_power - frame_median).median(dim=1, keepdim=True).values
    ).clamp_min(1.0e-6)
    burst_duty = (
        frame_log_power > frame_median + 1.5 * 1.4826 * frame_mad
    ).float().mean(dim=1)
    log_total_power = torch.log(token_band_power.sum(dim=1).clamp_min(1.0e-20))

    result = torch.cat(
        (
            relative_log_band,
            torch.stack(
                (
                    log_total_power,
                    occupied_width,
                    entropy,
                    centroid,
                    bandwidth,
                    flatness,
                    peak_position,
                    peak_dominance,
                    occupied_fraction,
                    hop_rate,
                    hop_mean,
                    hop_q90,
                    hop_max,
                    unique_fraction,
                    dwell_mean,
                    dwell_max,
                    flux_mean,
                    flux_q90,
                    flux_max,
                    frame_log_std,
                    frame_log_range,
                    burst_duty,
                ),
                dim=1,
            ),
        ),
        dim=1,
    ).float()
    output = result.cpu().numpy()
    if output.shape != (token_count, len(TOKEN_FEATURE_NAMES)):
        raise RuntimeError(f"spectral-activity feature shape drift: {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("spectral-activity features contain non-finite values")
    return output


def _metadata(rows: tuple[Any, ...]) -> dict[str, np.ndarray]:
    return {
        "event": np.asarray([row.event for row in rows]),
        "identity": np.asarray([row.identity for row in rows]),
        "behavior": np.asarray([row.behavior for row in rows]),
        "class_index": np.asarray([row.class_index for row in rows], dtype=np.int64),
        "condition": np.asarray([row.condition for row in rows], dtype=np.int64),
        "class_names": np.asarray(BEHAVIOR5_CLASSES),
        "token_feature_names": np.asarray(TOKEN_FEATURE_NAMES),
        "sample_rate_hz": np.asarray(SAMPLE_RATE, dtype=np.int64),
        "nfft": np.asarray(NFFT, dtype=np.int64),
        "frames_per_token": np.asarray(FRAMES_PER_TOKEN, dtype=np.int64),
        "token_samples": np.asarray(TOKEN_SAMPLES, dtype=np.int64),
        "tokens": np.asarray(TOKENS, dtype=np.int64),
        "trailing_samples_discarded": np.asarray(TRAILING_SAMPLES, dtype=np.int64),
        "schema": np.asarray(SCHEMA),
    }


def _open_partial(
    directory: Path, metadata: dict[str, np.ndarray],
) -> tuple[np.memmap, np.memmap]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in metadata.items():
        path = directory / f"{name}.npy"
        if not path.exists():
            np.save(path, expected, allow_pickle=False)
        if not np.array_equal(np.load(path, allow_pickle=False), expected):
            raise RuntimeError(f"partial metadata mismatch: {name}")
    trajectory_path = directory / "trajectory.npy"
    completed_path = directory / "completed.npy"
    if not trajectory_path.exists():
        value = np.lib.format.open_memmap(
            trajectory_path, mode="w+", dtype=np.float32,
            shape=(EVENTS, TOKENS, len(TOKEN_FEATURE_NAMES)),
        )
        value[:] = np.nan
        value.flush()
        del value
    if not completed_path.exists():
        value = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.bool_, shape=(EVENTS,),
        )
        value[:] = False
        value.flush()
        del value
    trajectory = np.lib.format.open_memmap(trajectory_path, mode="r+")
    completed = np.lib.format.open_memmap(completed_path, mode="r+")
    expected_shape = (EVENTS, TOKENS, len(TOKEN_FEATURE_NAMES))
    if trajectory.shape != expected_shape or completed.shape != (EVENTS,):
        raise RuntimeError("spectral-activity partial-cache shape drift")
    return trajectory, completed


def _validate_output(path: Path, metadata: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as cache:
        for name, expected in metadata.items():
            if name not in cache.files or not np.array_equal(cache[name], expected):
                raise RuntimeError(f"spectral-activity metadata mismatch: {name}")
        expected_shape = (EVENTS, TOKENS, len(TOKEN_FEATURE_NAMES))
        if cache["trajectory"].shape != expected_shape:
            raise RuntimeError("spectral-activity trajectory shape mismatch")
        if not np.isfinite(cache["trajectory"]).all():
            raise RuntimeError("spectral-activity trajectory contains non-finite values")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows = load_drff_r2_dataset1_behavior5_inventory(args.dataset_root)
    metadata = _metadata(rows)
    if args.output.exists():
        _validate_output(args.output, metadata)
        return {"status": "existing", "path": str(args.output.resolve())}
    trajectory, completed = _open_partial(args.partial_dir, metadata)
    resumed = int(completed.sum())
    processed_now = 0
    device = _device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for event_index, row in enumerate(rows):
        if bool(completed[event_index]):
            continue
        event_started = time.perf_counter()
        with h5py.File(row.path, "r") as source:
            i_data, q_data = source["RF0_I"], source["RF0_Q"]
            if i_data.shape != (RECORDING_SAMPLES, 1) or q_data.shape != i_data.shape:
                raise RuntimeError(f"unexpected recording shape: {row.event}")
            for first_token in range(0, TOKENS, args.tokens_per_read):
                token_count = min(args.tokens_per_read, TOKENS - first_token)
                first_sample = first_token * TOKEN_SAMPLES
                last_sample = first_sample + token_count * TOKEN_SAMPLES
                real = np.asarray(i_data[first_sample:last_sample, 0], dtype=np.float32)
                imag = np.asarray(q_data[first_sample:last_sample, 0], dtype=np.float32)
                trajectory[
                    event_index, first_token : first_token + token_count
                ] = spectral_activity_tokens(real, imag, device)
        if not np.isfinite(trajectory[event_index]).all():
            raise RuntimeError(f"incomplete spectral trajectory: {row.event}")
        trajectory.flush()
        completed[event_index] = True
        completed.flush()
        processed_now += 1
        complete_count = int(completed.sum())
        print(
            f"spectral-activity {complete_count}/{EVENTS} {row.event} "
            f"event_seconds={time.perf_counter() - event_started:.2f}", flush=True,
        )
        if args.max_events and processed_now >= args.max_events:
            break
    complete_count = int(completed.sum())
    if complete_count != EVENTS:
        return {
            "status": "partial", "completed_events": complete_count,
            "partial_dir": str(args.partial_dir.resolve()),
            "device": str(device),
            "peak_device_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "seconds": time.perf_counter() - started,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **metadata, trajectory=np.asarray(trajectory))
    os.replace(temporary, args.output)
    _validate_output(args.output, metadata)
    return {
        "status": "complete", "path": str(args.output.resolve()),
        "events": EVENTS, "tokens_per_event": TOKENS,
        "token_duration_ms": 1_000.0 * TOKEN_SAMPLES / SAMPLE_RATE,
        "device": str(device),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "resumed_events": resumed, "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root", nargs="?", type=Path,
        default=_DATASET_ROOT,
    )
    parser.add_argument(
        "--partial-dir", type=Path,
        default=WORKSPACE / "tmp/drff_r2_dataset1_behavior5_spectral_activity_v1.partial",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "tmp/drff_r2_dataset1_behavior5_spectral_activity_v1.npz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokens-per-read", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()
    if args.tokens_per_read < 1:
        raise ValueError("tokens-per-read must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))
