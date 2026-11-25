"""A stand-in reference classifier for running VET end to end.

VET keeps a source-trained reference classifier as its default answer (Step 1). The paper's reference is the
authors' hierarchical classifier of Section IV-B, whose outputs give the paper's numbers; it is not part of this
package. `make_logistic_reference` builds a simple reference with the required interface: standardization, PCA, and a
class-balanced logistic regression on the given features. Expect different numbers from the paper with it.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_reference(x: np.ndarray, y: np.ndarray, components: int = 32, seed: int = 0,
                            temperature: float = 1.0):
    """reference(train, test) -> (h, d) fitted on x[train], y[train] (four-state labels).

    h is the arg-max state; d is the class posterior softened by `temperature` (> 1 spreads it), which matters because
    the anchored candidate can only overturn h where d leaves mass on other states.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)

    def reference(train: np.ndarray, test: np.ndarray):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(components, x.shape[1], len(train) - 1), random_state=seed)),
            ("logistic", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
        ]).fit(x[train], y[train])
        logit = np.full((len(test), 4), -50.0)
        logit[:, model.classes_] = np.log(np.clip(model.predict_proba(x[test]), 1e-12, 1.0)) / temperature
        d = np.exp(logit - logit.max(1, keepdims=True))
        d /= d.sum(1, keepdims=True)
        return d.argmax(1), d

    return reference
