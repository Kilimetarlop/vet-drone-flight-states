"""Recording views of the chained state posterior (paper Section IV-C).

Envelope view (454 dimensions). Frames of 16384 samples (8544 complete frames per 1.4 s recording) give four log-power
traces: the envelope P(tau) (log mean power) and the log of the frame spread, peak, and peak-to-mean ratio. Each trace
is standardized by its median and median absolute deviation within the recording, then pooled over five time scales
(distribution and difference summaries), eight temporal bins, trends, burst statistics at four thresholds, and
periodicity (autocorrelation lags and peaks, 16 period bands, spectral shape).

STFT view (1524 dimensions). 64 windows of 16384 samples spread uniformly over the recording are stored as float16;
each is centred and normalized to unit power. A 512-point STFT with a Hann window and hop 512 gives 32 frames; the 96
central bins give four channels: log magnitude, its frame change, and the sine and cosine of the frame-to-frame phase
change. Pooling over bands, windows, a temporal cosine transform, and statistics gives the 1524-dimensional view.

Both are label-free. The code is the code that produced the paper's caches (Section V-A recordings).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import RECORDING_SAMPLES, read_iq

# ---------------------------------------------------------------------------------------------------- envelope view
FRAME_SAMPLES = 16_384
FRAMES = RECORDING_SAMPLES // FRAME_SAMPLES                     # 8544; the final 15104 samples are discarded
SCALES = (1, 4, 16, 64, 256)
TEMPORAL_BINS = 8
ACF_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
ACF_RANGES = ((2, 8), (8, 32), (32, 128), (128, 512), (512, 2048))
BURST_THRESHOLDS = (0.5, 1.0, 2.0, 3.0)
_EPSILON = np.float32(1.0e-20)


def envelope_frames(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """(frames, 4) log mean power, log power std, log peak power, log peak-to-mean power per 16384-sample frame."""
    real = np.asarray(real, dtype=np.float32).reshape(-1, FRAME_SAMPLES)
    imag = np.asarray(imag, dtype=np.float32).reshape(-1, FRAME_SAMPLES)
    power = np.square(real, dtype=np.float32)
    power += np.square(imag, dtype=np.float32)
    mean_power = power.mean(axis=1, dtype=np.float32)
    std_power = power.std(axis=1, dtype=np.float32)
    peak_power = power.max(axis=1)
    return np.stack((np.log(np.maximum(mean_power, _EPSILON)), np.log(np.maximum(std_power, _EPSILON)),
                     np.log(np.maximum(peak_power, _EPSILON)),
                     np.log(np.maximum(peak_power, _EPSILON) / np.maximum(mean_power, _EPSILON))),
                    axis=1).astype(np.float32)


def envelope_trajectory(path: str | Path, frames_per_read: int = 512) -> np.ndarray:
    """(8544, 4) envelope traces of one recording file."""
    out = np.empty((FRAMES, 4), dtype=np.float32)
    for first in range(0, FRAMES, frames_per_read):
        count = min(frames_per_read, FRAMES - first)
        real, imag = read_iq(path, first * FRAME_SAMPLES, count * FRAME_SAMPLES)
        out[first:first + count] = envelope_frames(real, imag)
    return out


def _distribution(v):
    q = np.quantile(v, (0.10, 0.25, 0.50, 0.75, 0.90))
    return [v.mean(), v.std(), *q, q[3] - q[1], q[4] - q[0]]


def _difference(v):
    dv = np.diff(v)
    a = np.abs(dv)
    return [dv.mean(), dv.std(), a.mean(), np.quantile(a, 0.50), np.quantile(a, 0.90), a.max()]


def _run_lengths(mask):
    edge = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return (np.flatnonzero(edge == -1) - np.flatnonzero(edge == 1)).astype(np.float64)


def _bursts(v):
    median = float(np.median(v))
    scale = max(1.4826 * float(np.median(np.abs(v - median))), 1.0e-6)
    z = (v - median) / scale
    out, n = [], float(len(v))
    for thr in BURST_THRESHOLDS:
        active = z > thr
        runs, gaps = _run_lengths(active), _run_lengths(~active)
        runs = runs if len(runs) else np.zeros(1)
        gaps = gaps if len(gaps) else np.zeros(1)
        out += [active.mean(), np.count_nonzero(np.diff(active)) / max(len(v) - 1, 1), len(runs) / n,
                runs.mean() / n, np.quantile(runs, 0.90) / n, runs.max() / n, gaps.mean() / n,
                np.quantile(gaps, 0.90) / n, gaps.max() / n]
    out += [float(np.mean(z < -thr)) for thr in BURST_THRESHOLDS]
    return out


def _periodicity(v):
    c = np.asarray(v, dtype=np.float64) - np.median(v)
    c /= max(1.4826 * np.median(np.abs(c)), 1.0e-6)
    size = 1 << (2 * len(c) - 1).bit_length()
    acf = np.fft.irfft(np.square(np.abs(np.fft.rfft(c, n=size))), n=size)[:len(c)]
    acf /= max(float(acf[0]), 1.0e-12)
    out = [float(acf[lag]) for lag in ACF_LAGS]
    for first, last in ACF_RANGES:
        region = acf[first:last]
        k = int(np.argmax(region))
        out += [float(region[k]), float((first + k) / len(v))]
    power = np.square(np.abs(np.fft.rfft(c)))[1:]
    total = max(float(power.sum()), 1.0e-12)
    edges = np.unique(np.clip(np.rint(np.geomspace(1, len(power) + 1, 17)).astype(int), 1, len(power) + 1))
    out += [float(power[a - 1:b - 1].sum() / total) for a, b in zip(edges[:-1], edges[1:])]     # 16 period bands
    p = power / total
    entropy = -float(np.sum(p * np.log(np.maximum(p, 1.0e-20)))) / np.log(len(power))
    k = int(np.argmax(power))
    freq = np.arange(1, len(power) + 1, dtype=np.float64) / len(v)
    return out + [entropy, float(freq[k]), float(power[k] / total), float(np.sum(p * freq))]


def _pooled(trajectory: np.ndarray) -> np.ndarray:
    v = np.asarray(trajectory, dtype=np.float64)
    out = []
    for scale in SCALES:
        n = len(v) // scale
        pooled = v[:n * scale].reshape(n, scale, v.shape[1]).mean(axis=1)
        for ch in range(v.shape[1]):
            out += _distribution(pooled[:, ch]) + _difference(pooled[:, ch])
    edges = np.linspace(0, len(v), TEMPORAL_BINS + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        for ch in range(v.shape[1]):
            out += [float(v[a:b, ch].mean()), float(v[a:b, ch].std())]
    t = np.linspace(-1.0, 1.0, len(v))
    for ch in range(v.shape[1]):
        out += [float(np.dot(t, v[:, ch] - v[:, ch].mean()) / float(np.dot(t, t))),
                float(v[edges[-2]:edges[-1], ch].mean() - v[edges[0]:edges[1], ch].mean())]
    out += _bursts(v[:, 0]) + _periodicity(v[:, 0])
    return np.asarray(out, dtype=np.float32)


def robust_standardize(trajectory: np.ndarray) -> np.ndarray:
    """Median and MAD standardization of each trace within the recording, clipped to +-50."""
    median = np.median(trajectory, axis=0, keepdims=True)
    mad = np.median(np.abs(trajectory - median), axis=0, keepdims=True)
    return np.clip((trajectory - median) / np.maximum(1.4826 * mad, 1.0e-5), -50.0, 50.0).astype(np.float32)


def envelope_view(trajectory: np.ndarray) -> np.ndarray:
    """phi_env: the 454-dimensional envelope view of one (8544, 4) trajectory."""
    return _pooled(robust_standardize(trajectory))


# -------------------------------------------------------------------------------------------------------- STFT view
WINDOWS = 64
NFFT, HOP, STFT_FRAMES, BINS = 512, 512, 32, 96


def uniform_windows(path: str | Path) -> np.ndarray:
    """(64, 16384, 2) float16 I/Q windows starting at 64 uniformly spaced positions that span the recording."""
    starts = np.linspace(0, RECORDING_SAMPLES - FRAME_SAMPLES, WINDOWS, dtype=np.int64)
    out = np.empty((WINDOWS, FRAME_SAMPLES, 2), dtype=np.float16)
    for k, start in enumerate(starts.tolist()):
        real, imag = read_iq(path, start, FRAME_SAMPLES)
        out[k] = np.stack((real, imag), axis=1).astype(np.float16)
    return out


def _standardize(t):
    return (t - t.mean(dim=(-2, -1), keepdim=True)) / t.std(dim=(-2, -1), keepdim=True,
                                                             unbiased=False).clamp_min(1.0e-5)


def stft_channels(windows: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(64, 4, 32, 96) float16 channels: log magnitude, its frame change, sine and cosine of the phase change."""
    import torch

    with torch.inference_mode():
        raw = torch.as_tensor(np.asarray(windows)[None], device=device).float()
        raw -= raw.mean(dim=2, keepdim=True)
        raw /= torch.sqrt(raw.square().mean(dim=(2, 3), keepdim=True)).clamp_min(1.0e-6)
        flat = raw.reshape(-1, raw.shape[2], 2)
        spectrum = torch.stft(torch.complex(flat[..., 0], flat[..., 1]), n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                              window=torch.hann_window(NFFT, device=device), center=False, onesided=False,
                              return_complex=True)
        spectrum = torch.fft.fftshift(spectrum, dim=1)[:, NFFT // 2 - BINS // 2:NFFT // 2 + BINS // 2]
        log_mag = torch.log1p(spectrum.abs())
        delta = torch.zeros_like(log_mag)
        delta[:, :, 1:] = log_mag[:, :, 1:] - log_mag[:, :, :-1]
        phase = torch.angle(spectrum[:, :, 1:] * spectrum[:, :, :-1].conj())
        sin, cos = torch.zeros_like(log_mag), torch.ones_like(log_mag)
        sin[:, :, 1:], cos[:, :, 1:] = torch.sin(phase), torch.cos(phase)
        feature = torch.stack((_standardize(log_mag), _standardize(delta), sin, cos), dim=1).permute(0, 1, 3, 2)
        return feature.reshape(len(windows), 4, STFT_FRAMES, BINS).half().cpu().numpy()


def stft_view(channels: np.ndarray) -> np.ndarray:
    """phi_stft: the 1524-dimensional STFT view of one (64, 4, 32, 96) channel array."""
    from scipy.fft import dct

    v = np.asarray(channels, dtype=np.float32)
    mag, delta, sin, cos = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    frequency = np.concatenate((mag.mean(axis=(0, 1)), mag.std(axis=(0, 1)), np.abs(delta).mean(axis=(0, 1)),
                                np.sqrt(np.square(sin.mean(axis=(0, 1))) + np.square(cos.mean(axis=(0, 1))))))
    token = np.concatenate((mag.mean(axis=1).reshape(len(v), 12, 8).mean(axis=2), mag.std(axis=(1, 2))[:, None],
                            np.abs(delta).mean(axis=(1, 2))[:, None],
                            np.sqrt(np.square(sin.mean(axis=(1, 2))) + np.square(cos.mean(axis=(1, 2))))[:, None]),
                           axis=1)
    temporal = dct(token, type=2, axis=0, norm="ortho")[:8]
    statistics = np.concatenate((token.mean(axis=0), token.std(axis=0), np.quantile(token, 0.10, axis=0),
                                 np.quantile(token, 0.90, axis=0)))
    return np.concatenate((frequency, token.ravel(), temporal.ravel(), statistics)).astype(np.float32)


def recording_views(path: str | Path, device: str = "cpu") -> dict[str, np.ndarray]:
    """Both views of one recording file: {'envelope': (454,), 'stft': (1524,)}."""
    return {"envelope": envelope_view(envelope_trajectory(path)),
            "stft": stft_view(stft_channels(uniform_windows(path), device))}
