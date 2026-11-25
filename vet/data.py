"""DRFF-R2 Dataset 1 inventory (paper Section V-A and Table I).

Dataset 1 holds one 1.4 s complex-baseband recording (U1-D2: 5.745 GHz, 100 MS/s) per product and flight condition:
eight DJI products x 26 conditions = 208 MATLAB v7.3 files named like ``mavic3_1_Cruise_c2_u1_d2.mat`` with datasets
``RF0_I`` and ``RF0_Q`` of shape (140000000, 1). Conditions c1-c21 map to the four flight states; the five shading
conditions c22-c26 are excluded, leaving 168 recordings. The dataset is not redistributed here; obtain it from its
providers (Zheng et al., Scientific Data, 2026, doi:10.1038/s41597-026-08490-8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRODUCTS = ("mavic3_1", "mavic3C_1", "mavic3S_1", "mavicAir2_1", "mavicAir2s_1", "mini3pro_2", "mini4PRO_1",
            "mini5PRO_1")
PRODUCT_NAMES = {"mavic3_1": "Mavic 3", "mavic3C_1": "Mavic 3C", "mavic3S_1": "Mavic 3S", "mavicAir2_1": "Mavic Air 2",
                 "mavicAir2s_1": "Mavic Air 2S", "mini3pro_2": "Mini 3 Pro", "mini4PRO_1": "Mini 4 Pro",
                 "mini5PRO_1": "Mini 5 Pro"}
SOURCE_STATE = {**{c: "Cruise" for c in range(1, 16)}, 16: "TakeOff", 17: "Ascend", 18: "Ascend", 19: "Descend",
                20: "Descend", 21: "Landing", **{c: "Shading" for c in range(22, 27)}}
# Five recorded classes of c1-c21 (straight cruise, curved cruise, hover, ascend, descend) and the four states.
FIVE_CLASS = {**{c: 0 for c in range(1, 7)}, **{c: 1 for c in range(7, 13)}, **{c: 2 for c in range(13, 16)},
              **{c: 3 for c in range(16, 19)}, **{c: 4 for c in range(19, 22)}}
HEIGHT_M = {**{c: (10, 30, 50)[(c - 1) % 6 // 2] for c in range(1, 13)}, 13: 10, 14: 30, 15: 50}
SPEED_MS = {**{c: ("0-5", "5-10")[(c - 1) % 2] for c in range(1, 13)}, 17: "0-5", 18: "5-10", 19: "0-5", 20: "5-10"}
RECORDING_SAMPLES = 140_000_000
SAMPLE_RATE = 100_000_000
_FILE = re.compile(r"^(?P<product>[A-Za-z0-9]+_[0-9]+)_(?P<state>[A-Za-z]+)_c(?P<condition>[0-9]+)_u1_d2\.mat$")


@dataclass(frozen=True)
class Recording:
    product: str
    condition: int
    path: Path

    @property
    def event(self) -> str:
        return self.path.name

    @property
    def five_class(self) -> int:
        return FIVE_CLASS[self.condition]

    @property
    def state(self) -> int:
        return (0, 0, 1, 2, 3)[FIVE_CLASS[self.condition]]


def inventory(root: str | Path) -> list[Recording]:
    """The 168 recordings of c1-c21, ordered by product (PRODUCTS order) and condition. Requires all 208 files."""
    root = Path(root)
    expected = {f"{p}_{SOURCE_STATE[c]}_c{c}_u1_d2.mat" for p in PRODUCTS for c in range(1, 27)}
    present = {p.name for p in root.glob("*.mat")}
    missing = sorted(expected - present)
    if missing:
        raise FileNotFoundError(f"{len(missing)} Dataset 1 files missing under {root}, e.g. {missing[:3]}")
    rows = []
    for name in expected:
        m = _FILE.fullmatch(name)
        c = int(m.group("condition"))
        if c <= 21:
            rows.append(Recording(m.group("product"), c, root / name))
    return sorted(rows, key=lambda r: (PRODUCTS.index(r.product), r.condition))


def read_iq(path: str | Path, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """I and Q samples [start, start + count) as float32 arrays."""
    import h5py

    with h5py.File(path, "r") as f:
        i_data, q_data = f["RF0_I"], f["RF0_Q"]
        return (np.asarray(i_data[start:start + count, 0], dtype=np.float32),
                np.asarray(q_data[start:start + count, 0], dtype=np.float32))
