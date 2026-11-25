"""VET end to end on a synthetic pool (no data needed).

Eight synthetic "products" each record 21 recordings with the state mix of DRFF-R2 (12 Horizontal, 3 Hover, 3 Ascend,
3 Descend). Every product shifts its features in its own way, and the reference classifier mistakes slow horizontal
transit for hover on some products, the failure that VET's direction test is built to verify. Numbers printed here
describe this synthetic pool only.

    python examples/demo_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vet import metrics  # noqa: E402
from vet.model import run_whole_product_holdout  # noqa: E402
from vet.reference import make_logistic_reference  # noqa: E402
from vet.states import STATES, direction_name  # noqa: E402


def synthetic_pool(seed: int = 7):
    rng = np.random.default_rng(seed)
    states = np.array([0] * 12 + [1] * 3 + [2] * 3 + [3] * 3)
    slow = np.array([1, 0] * 6 + [0] * 9, dtype=bool)          # half of the horizontal recordings are slow transit
    y, product, env, stft, ref = [], [], [], [], []
    for p in range(8):
        shift_env, shift_stft = rng.normal(0, 1.2, 40), rng.normal(0, 1.2, 60)
        shift_ref = rng.normal(0, 1.2, 30)
        for s, is_slow in zip(states, slow):
            motion = 0.0 if s == 1 else (1.2 if is_slow else 2.0)            # the envelope separates slow transit
            vertical = 1.8 if s in (2, 3) else 0.0
            ascent = 1.5 if s == 2 else (-1.5 if s == 3 else 0.0)
            e = rng.normal(0, 1, 40) + shift_env
            e[:8] += motion
            e[8:16] += vertical
            f = rng.normal(0, 1, 60) + shift_stft
            f[:10] += ascent
            f[10:20] += 0.5 * vertical
            r = rng.normal(0, 1, 30) + shift_ref                            # the reference has its own features:
            r[:6] += 1.6 if (s == 0 and not is_slow) else 0.0               # slow transit looks exactly like hover
            r[6:12] += vertical
            r[12:18] += ascent
            y.append(s), product.append(f"product_{p}"), env.append(e), stft.append(f), ref.append(r)
    return (np.array(y), np.array(product), {"envelope": np.array(env), "stft": np.array(stft)}, np.array(ref))


def main() -> None:
    y, product, views, ref_features = synthetic_pool()
    reference = make_logistic_reference(ref_features, y, components=16, temperature=3.0)
    res = run_whole_product_holdout(y, product, views, reference)
    m_ref, m_vet = metrics.scores(y, res.reference), metrics.scores(y, res.report)
    print(f"reference  accuracy {m_ref['accuracy']:.3f}  balanced accuracy {m_ref['balanced_accuracy']:.3f}")
    print(f"VET        accuracy {m_vet['accuracy']:.3f}  balanced accuracy {m_vet['balanced_accuracy']:.3f}")
    print("paired outcomes of VET against the reference:", metrics.paired(y, res.reference, res.report))
    for f in res.folds:
        if f.selection is None:
            print(f"  {f.target}: no eligible configuration; the reference is reported")
            continue
        admitted = [direction_name(k, l) for k, l in zip(*np.nonzero(f.selection.admitted))]
        print(f"  {f.target}: lambda={f.selection.lam:.2f} eta={f.selection.eta:.2f} admitted {admitted}")
    print("states:", ", ".join(f"{i}={s}" for i, s in enumerate(STATES)))


if __name__ == "__main__":
    main()
