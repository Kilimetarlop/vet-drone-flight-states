"""Reference posterior, chained state posterior, and anchored candidate (paper eqs. (4), (6)-(8)).

All posteriors are arrays of shape (n, 4) in the order of ``vet.states.STATES``.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1.0e-8                                          # floor of every probability (Section IV-D)
LAMBDAS = (1.0 / 3.0, 2.0 / 5.0)                      # chain weight lambda
ETAS = (math.log(3.0), math.log(4.0))                 # anchor eta: odds of the judged state times three or four
CONFIGURATIONS = tuple((lam, eta) for lam in LAMBDAS for eta in ETAS)   # Theta, four configurations


def normalise(p: np.ndarray) -> np.ndarray:
    """Floor at EPS and renormalise each row."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def _softmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return normalise(np.exp(np.clip(s - s.max(axis=1, keepdims=True), -80.0, 0.0)))


def _one_hot(k: np.ndarray) -> np.ndarray:
    return np.eye(4)[np.asarray(k, dtype=np.int64)]


def reference_posterior(q5: np.ndarray) -> np.ndarray:
    """Eq. (4): five-class posterior [straight, curved, hover, ascend, descend] -> four-state posterior d."""
    q5 = np.asarray(q5, dtype=np.float64)
    return normalise(np.column_stack((q5[:, 0] + q5[:, 1], q5[:, 2:])))


def chained_posterior(scores: np.ndarray) -> np.ndarray:
    """Eq. (6): scores [s, v, u] of the three heads -> chained state posterior r.

    s estimates Pr(Hover | x); v estimates Pr(Ascend or Descend | moving, x); u estimates Pr(Ascend | vertical, x).
    """
    s, v, u = np.clip(np.asarray(scores, dtype=np.float64), EPS, 1.0 - EPS).T
    moving = 1.0 - s
    return normalise(np.column_stack((moving * (1.0 - v), s, moving * v * u, moving * v * (1.0 - u))))


def anchor(d: np.ndarray, h: np.ndarray, eta: float) -> np.ndarray:
    """Eq. (7): anchored reference posterior b; multiplies the odds of the judged state h by exp(eta)."""
    return _softmax(np.log(np.clip(d, EPS, 1.0)) + eta * _one_hot(h))


def blend(r: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Eq. (8): candidate posterior z = lam * r + (1 - lam) * b."""
    return normalise(lam * np.asarray(r) + (1.0 - lam) * np.asarray(b))


def candidate_posterior(d: np.ndarray, h: np.ndarray, scores: np.ndarray, lam: float, eta: float) -> np.ndarray:
    """Steps 2 and 3: z_i under configuration theta = (lam, eta); the candidate state is z.argmax(1)."""
    return blend(chained_posterior(scores), anchor(d, h, eta), lam)
