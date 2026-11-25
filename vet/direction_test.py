"""Correction outcomes and the direction test (paper eqs. (3), (9)-(11)).

A change replaces the judged state h_i by the candidate c_i != h_i. Against the label y_i it is a repair (c_i = y_i),
a damage (h_i = y_i), or a neutral change (both wrong). The direction test admits the ordered change k -> l when, on
held-out source predictions, its repairs repeat (R >= r_min), exceed its damages (R > D), and make up a share of at
least ``share`` of its changes (R >= share * |C|).
"""

from __future__ import annotations

import numpy as np

R_MIN = 2
SHARE = 2.0 / 3.0


def outcomes(h: np.ndarray, c: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eq. (3) per recording: 'repair', 'damage', 'neutral', or '' when c == h."""
    h, c, y = (np.asarray(a, dtype=np.int64) for a in (h, c, y))
    out = np.full(len(h), "", dtype=object)
    changed = c != h
    out[changed & (c == y)] = "repair"
    out[changed & (h == y)] = "damage"
    out[changed & (h != y) & (c != y)] = "neutral"
    return out


def direction_counts(h: np.ndarray, c: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Eq. (9): for every ordered pair (k, l), the size |C_kl| and the repairs R_kl and damages D_kl."""
    h, c, y = (np.asarray(a, dtype=np.int64) for a in (h, c, y))
    n = np.zeros((4, 4), dtype=np.int64)
    rep = np.zeros((4, 4), dtype=np.int64)
    dam = np.zeros((4, 4), dtype=np.int64)
    changed = c != h
    np.add.at(n, (h[changed], c[changed]), 1)
    np.add.at(rep, (h[changed & (c == y)], c[changed & (c == y)]), 1)
    np.add.at(dam, (h[changed & (h == y)], c[changed & (h == y)]), 1)
    return {"changes": n, "repairs": rep, "damages": dam, "neutral": n - rep - dam}


def admitted_directions(h: np.ndarray, c: np.ndarray, y: np.ndarray, r_min: int = R_MIN,
                        share: float = SHARE) -> np.ndarray:
    """Eq. (10): boolean (4, 4) matrix of admitted ordered changes (row: judged state, column: candidate)."""
    cnt = direction_counts(h, c, y)
    n, rep, dam = cnt["changes"], cnt["repairs"], cnt["damages"]
    ratio = np.divide(rep, n, out=np.zeros((4, 4)), where=n > 0)
    admitted = (rep >= r_min) & (rep > dam) & (ratio >= share - 1e-12)
    np.fill_diagonal(admitted, False)
    return admitted


def decide(h: np.ndarray, c: np.ndarray, admitted: np.ndarray) -> np.ndarray:
    """Eq. (11): report c_i when its change h_i -> c_i is admitted, otherwise keep h_i."""
    h, c = np.asarray(h, dtype=np.int64), np.asarray(c, dtype=np.int64)
    ok = np.asarray(admitted, dtype=bool)[h, c] & (c != h)
    out = h.copy()
    out[ok] = c[ok]
    return out
