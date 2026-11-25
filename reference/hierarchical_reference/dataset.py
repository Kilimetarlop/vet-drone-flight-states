"""DRFF-R2 Dataset 1 inventory: the recordings of conditions c1-c21, their five recorded classes, and the
excluded shading recordings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final


DATASET1_IDENTITIES: Final[tuple[str, ...]] = (
    "mavic3_1",
    "mavic3C_1",
    "mavic3S_1",
    "mavicAir2_1",
    "mavicAir2s_1",
    "mini3pro_2",
    "mini4PRO_1",
    "mini5PRO_1",
)


BEHAVIOR5_CLASSES: Final[tuple[str, ...]] = (
    "LINEAR_CRUISE",
    "CURVED_TURN",
    "HOVER",
    "TAKEOFF_ASCEND",
    "DESCEND_LAND",
)


BEHAVIOR5_CONDITIONS: Final[dict[str, tuple[int, ...]]] = {
    "LINEAR_CRUISE": tuple(range(1, 7)),
    "CURVED_TURN": tuple(range(7, 13)),
    "HOVER": tuple(range(13, 16)),
    "TAKEOFF_ASCEND": tuple(range(16, 19)),
    "DESCEND_LAND": tuple(range(19, 22)),
}


EXPECTED_BEHAVIOR5_CLASS_COUNTS: Final[dict[str, int]] = {
    name: len(BEHAVIOR5_CONDITIONS[name]) * len(DATASET1_IDENTITIES)
    for name in BEHAVIOR5_CLASSES
}


_CLASS_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(BEHAVIOR5_CLASSES)
}


_EXPECTED_SOURCE_STATE: Final[dict[int, str]] = {
    **{condition: "Cruise" for condition in range(1, 16)},
    16: "TakeOff",
    17: "Ascend",
    18: "Ascend",
    19: "Descend",
    20: "Descend",
    21: "Landing",
    **{condition: "Shading" for condition in range(22, 27)},
}


_BEHAVIOR_BY_CONDITION: Final[dict[int, str]] = {
    condition: behavior
    for behavior, conditions in BEHAVIOR5_CONDITIONS.items()
    for condition in conditions
}


_DATASET1_FILE = re.compile(
    r"^(?P<identity>[A-Za-z0-9]+_[0-9]+)_"
    r"(?P<state>Cruise|TakeOff|Ascend|Descend|Landing|Shading)_"
    r"c(?P<condition>[0-9]+)_u(?P<receiver>[0-9]+)_d(?P<day>[0-9]+)\.mat$"
)


@dataclass(frozen=True)
class DRFFR2BehaviorRecording:
    """One metadata-only dataset1 recording.

    ``behavior`` and ``class_index`` are ``None`` only for the five Shading
    recordings per UAV.  Such rows are validated but are not returned by the
    five-class inventory loader.
    """

    identity: str
    source_state: str
    condition: int
    receiver_index: int
    day_index: int
    behavior: str | None
    class_index: int | None
    path: str

    @property
    def event(self) -> str:
        return Path(self.path).name

    @property
    def channel(self) -> str:
        return f"U{self.receiver_index}-D{self.day_index}"


def _source_state_for_condition(condition: int) -> str:
    try:
        return _EXPECTED_SOURCE_STATE[condition]
    except KeyError as exc:
        raise ValueError(
            f"dataset1 condition must be in c1--c26, observed c{condition}"
        ) from exc


def expected_dataset1_filenames() -> tuple[str, ...]:
    return tuple(
        f"{identity}_{_source_state_for_condition(condition)}_"
        f"c{condition}_u1_d2.mat"
        for identity in DATASET1_IDENTITIES
        for condition in range(1, 27)
    )


def parse_drff_r2_dataset1_recording(
    path: str | Path,
) -> DRFFR2BehaviorRecording:
    source = Path(path)
    match = _DATASET1_FILE.fullmatch(source.name)
    if match is None:
        raise ValueError(f"not an exact DRFF-R2 dataset1 filename: {source.name}")

    identity = match.group("identity")
    if identity not in DATASET1_IDENTITIES:
        raise ValueError(f"unexpected DRFF-R2 dataset1 identity: {identity}")
    condition = int(match.group("condition"))
    expected_state = _source_state_for_condition(condition)
    state = match.group("state")
    if state != expected_state:
        raise ValueError(
            f"dataset1 state/condition mismatch: c{condition} requires "
            f"{expected_state}, observed {state}"
        )
    receiver = int(match.group("receiver"))
    day = int(match.group("day"))
    if (receiver, day) != (1, 2):
        raise ValueError(
            "DRFF-R2 dataset1 five-behavior protocol requires U1-D2, "
            f"observed U{receiver}-D{day}"
        )

    behavior = _BEHAVIOR_BY_CONDITION.get(condition)
    return DRFFR2BehaviorRecording(
        identity=identity,
        source_state=state,
        condition=condition,
        receiver_index=receiver,
        day_index=day,
        behavior=behavior,
        class_index=None if behavior is None else _CLASS_INDEX[behavior],
        path=str(source.resolve()),
    )


def load_drff_r2_dataset1_behavior5_inventory(
    dataset_root: str | Path,
) -> tuple[DRFFR2BehaviorRecording, ...]:
    """Load the exact 168-recording five-class view of dataset1.

    This is intentionally fail-closed.  The directory must contain the exact
    208 MAT filenames (eight declared UAVs, every condition c1--c26), including
    the 40 excluded Shading files.  No HDF5 signal payload is opened here.
    """

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = tuple(sorted(root.glob("*.mat"), key=lambda path: path.name))
    observed_names = tuple(path.name for path in paths)
    expected_names = expected_dataset1_filenames()
    observed_folded = [name.casefold() for name in observed_names]
    if len(observed_folded) != len(set(observed_folded)):
        raise ValueError("dataset1 inventory contains case-insensitive duplicates")

    missing = sorted(set(expected_names) - set(observed_names))
    unexpected = sorted(set(observed_names) - set(expected_names))
    if missing or unexpected:
        missing_preview = ", ".join(missing[:3]) or "none"
        unexpected_preview = ", ".join(unexpected[:3]) or "none"
        raise ValueError(
            "dataset1 inventory mismatch: "
            f"expected=208 observed={len(observed_names)}; "
            f"missing=[{missing_preview}]; unexpected=[{unexpected_preview}]"
        )

    parsed = tuple(parse_drff_r2_dataset1_recording(path) for path in paths)
    cells = {(row.identity, row.condition) for row in parsed}
    expected_cells = {
        (identity, condition)
        for identity in DATASET1_IDENTITIES
        for condition in range(1, 27)
    }
    if cells != expected_cells or len(parsed) != 208:
        raise ValueError("dataset1 identity/condition inventory mismatch")

    identity_rank = {name: index for index, name in enumerate(DATASET1_IDENTITIES)}
    selected = tuple(
        sorted(
            (row for row in parsed if row.behavior is not None),
            key=lambda row: (identity_rank[row.identity], row.condition),
        )
    )
    counts = Counter(row.behavior for row in selected)
    if len(selected) != 168 or dict(counts) != EXPECTED_BEHAVIOR5_CLASS_COUNTS:
        raise ValueError(
            f"dataset1 behavior5 inventory drift: n={len(selected)}, "
            f"counts={dict(counts)}"
        )
    if any(
        row.class_index != _CLASS_INDEX[row.behavior]  # type: ignore[index]
        for row in selected
    ):
        raise AssertionError("dataset1 class-index mapping drift")
    return selected


import os as _os

# Folder with the 208 Dataset 1 .mat files; set the environment variable DRFF_R2_DATASET1.
DEFAULT_DATASET_ROOT = Path(_os.environ.get("DRFF_R2_DATASET1", "DRFF-R2/dataset1-single_drone_states"))
