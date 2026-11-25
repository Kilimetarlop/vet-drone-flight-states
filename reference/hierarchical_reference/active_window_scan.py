"""Scores uniformly spaced positions of a recording by in-band activity and selects the strongest,
well separated ones.
"""

from __future__ import annotations

import h5py
import numpy as np


SCAN_SAMPLES = 8_192


WINDOWS = 29


NMS_INDEX_RADIUS = 1


CENTRAL_BAND_FRACTION = 0.25


def _scan_scores(
    i_data: h5py.Dataset,
    q_data: h5py.Dataset,
    centers: np.ndarray,
    taper: np.ndarray,
) -> np.ndarray:
    values = []
    half = SCAN_SAMPLES // 2
    for center in centers:
        start = int(center) - half
        i = np.asarray(i_data[start : start + SCAN_SAMPLES, 0], dtype=np.float32)
        q = np.asarray(q_data[start : start + SCAN_SAMPLES, 0], dtype=np.float32)
        value = i.astype(np.complex64) + 1j * q.astype(np.complex64)
        value -= value.mean()
        values.append(value)
    batch = np.stack(values) * taper[None, :]
    spectrum = np.fft.fftshift(np.fft.fft(batch, axis=1), axes=1)
    power = np.abs(spectrum) ** 2
    width = int(SCAN_SAMPLES * CENTRAL_BAND_FRACTION)
    center = SCAN_SAMPLES // 2
    central = power[:, center - width // 2 : center + width // 2].sum(axis=1)
    return (central / power.sum(axis=1).clip(min=1.0e-12)).astype(np.float32)


def _select_nms(scores: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        index = int(candidate)
        if all(abs(index - prior) > NMS_INDEX_RADIUS for prior in selected):
            selected.append(index)
            if len(selected) == WINDOWS:
                break
    if len(selected) < WINDOWS:
        for candidate in np.argsort(scores)[::-1]:
            index = int(candidate)
            if index not in selected:
                selected.append(index)
                if len(selected) == WINDOWS:
                    break
    if len(selected) != WINDOWS or len(set(selected)) != WINDOWS:
        raise AssertionError("NMS failed to select 29 unique candidates")
    return np.asarray(sorted(selected), dtype=np.int64)
