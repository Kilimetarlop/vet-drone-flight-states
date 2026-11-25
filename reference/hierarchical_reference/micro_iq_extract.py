"""Extraction of the active micro-IQ windows of one recording.
"""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np

from .active_window_scan import SCAN_SAMPLES, _scan_scores, _select_nms


WINDOWS = 29


SAMPLES = 16_384


CANDIDATES = 256


def _extract_event(path: Path) -> tuple[np.ndarray, ...]:
    taper = np.hanning(SCAN_SAMPLES).astype(np.float32)
    with h5py.File(path, "r") as source:
        i_data, q_data = source["RF0_I"], source["RF0_Q"]
        length = int(i_data.shape[0])
        if q_data.shape != i_data.shape or length < SAMPLES:
            raise RuntimeError(f"invalid RF0_I/RF0_Q: {path.name}")
        candidates = np.linspace(
            SCAN_SAMPLES // 2,
            length - SCAN_SAMPLES // 2,
            CANDIDATES,
            dtype=np.int64,
        )
        scores = _scan_scores(i_data, q_data, candidates, taper)
        selected = _select_nms(scores)
        centers = candidates[selected]
        starts = np.clip(
            centers - SAMPLES // 2, 0, length - SAMPLES,
        ).astype(np.int64)
        iq = np.empty((WINDOWS, SAMPLES, 2), dtype=np.float16)
        for row, start in enumerate(starts.tolist()):
            real = np.asarray(i_data[start : start + SAMPLES, 0], dtype=np.float32)
            imag = np.asarray(q_data[start : start + SAMPLES, 0], dtype=np.float32)
            value = np.stack((real, imag), axis=1)
            if value.shape != (SAMPLES, 2) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid micro-IQ: {path.name}/{row}")
            if float(np.max(np.abs(value))) > np.finfo(np.float16).max:
                raise RuntimeError(f"float16 overflow: {path.name}/{row}")
            iq[row] = value.astype(np.float16)
    return selected, centers, starts, scores, iq
