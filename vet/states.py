"""Flight states and the map from the five recorded classes (paper Section III-A and Table I)."""

from __future__ import annotations

import numpy as np

STATES = ("Horizontal", "Hover", "Ascend", "Descend")
ABBREVIATIONS = ("Hor", "Hov", "Asc", "Des")
HORIZONTAL, HOVER, ASCEND, DESCEND = range(4)

# The reference classifier of the paper votes over five classes; both cruise classes map to Horizontal.
FIVE_CLASSES = ("straight cruise", "curved cruise", "hover", "ascend", "descend")
FIVE_TO_FOUR = np.array([0, 0, 1, 2, 3], dtype=np.int64)


def direction_name(k: int, l: int) -> str:
    """Ordered change of state k -> l, e.g. 'Hov->Hor'."""
    return f"{ABBREVIATIONS[k]}->{ABBREVIATIONS[l]}"
