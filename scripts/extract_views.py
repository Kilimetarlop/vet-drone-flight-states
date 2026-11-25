"""Extract the envelope view (454-d) and the STFT view (1524-d) of the 168 DRFF-R2 Dataset 1 recordings of the paper.

    python scripts/extract_views.py /path/to/DRFF-R2/dataset1-single_drone_states views.npz [--device cuda]

Each recording is about 0.5 GB; the full run reads 88 GB and took 12 minutes with --device cuda. The output holds, per
recording: event (file name), product, condition, four-state label, five-class label, and both views.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vet.data import inventory  # noqa: E402
from vet.views import recording_views  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="extract only the first N recordings (for a quick test)")
    args = ap.parse_args()
    rows = inventory(args.dataset_root)
    if args.limit:
        rows = rows[:args.limit]
    env, stft = np.zeros((len(rows), 454), np.float32), np.zeros((len(rows), 1524), np.float32)
    start = time.perf_counter()
    for i, r in enumerate(rows):
        v = recording_views(r.path, args.device)
        env[i], stft[i] = v["envelope"], v["stft"]
        print(f"{i + 1}/{len(rows)} {r.event} {time.perf_counter() - start:.0f}s", flush=True)
    np.savez_compressed(args.output, event=np.array([r.event for r in rows]),
                        product=np.array([r.product for r in rows]), condition=np.array([r.condition for r in rows]),
                        y=np.array([r.state for r in rows]), five_class=np.array([r.five_class for r in rows]),
                        envelope=env, stft=stft)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
