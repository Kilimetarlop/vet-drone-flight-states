"""VET: verified evidence transfer for radio-frequency flight states of unseen drones.

Modules follow the paper:
    states          flight states and the five-to-four class map            Section III-A
    data            DRFF-R2 Dataset 1 inventory and I/Q reading               Section V-A, Table I
    views           envelope view (454-d) and STFT view (1524-d)              Section IV-C
    heads           three conditional heads, eq. (5)                          Section IV-C
    candidate       reference posterior (4), chain (6), anchor (7), blend (8) Sections IV-B to IV-D
    direction_test  outcomes (3), change sets (9), admission (10), decision (11)
    configuration   eligibility (12) and lexicographic selection (13)         Section IV-F
    model           whole-product holdout, Algorithm 1, deployed monitor      Sections III-B, IV-F
    gate            confidence-gated correction baseline                      Section V-B
    analysis        held-out source analysis of the correction mechanism      Section V-F
    baselines       retrained domain-generalization baselines (PyTorch)       Section V-B
    metrics         accuracy, balanced accuracy, Macro-F1, paired tests       Sections III-D, V-G
"""

from .candidate import CONFIGURATIONS, anchor, blend, candidate_posterior, chained_posterior, reference_posterior
from .configuration import Selection, select_configuration
from .direction_test import admitted_directions, decide, direction_counts, outcomes
from .model import Monitor, fit_monitor, make_precomputed_reference, run_whole_product_holdout
from .states import ABBREVIATIONS, FIVE_TO_FOUR, STATES

__version__ = "1.0.0"
__all__ = ["CONFIGURATIONS", "anchor", "blend", "candidate_posterior", "chained_posterior", "reference_posterior",
           "Selection", "select_configuration", "admitted_directions", "decide", "direction_counts", "outcomes",
           "Monitor", "fit_monitor", "make_precomputed_reference", "run_whole_product_holdout", "ABBREVIATIONS",
           "FIVE_TO_FOUR", "STATES"]
