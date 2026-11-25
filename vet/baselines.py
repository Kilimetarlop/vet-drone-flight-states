"""Retrained domain-generalization baselines of the paper (Section V-B, Tables II and III). Requires PyTorch.

Every method reads the concatenated envelope and STFT views (1978 dimensions) and the four-state labels, with products
as domains: standardization, PCA to k in {16, 32} axes, a 32-unit ReLU layer and a linear classifier, class-balanced
cross-entropy, Adam (learning rate 1e-2, weight decay 1e-4), 200 full-batch epochs, penalties from epoch 50. Each fold
selects (k, penalty) by leave-one-product-out Macro-F1 on its source products (seed 0) and averages the probabilities
of three seeds on the target.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .metrics import scores

METHODS = {
    "ERM": (1e-4, 1e-3, 1e-2),          # weight decay
    "IRM": (1.0, 10.0, 100.0),          # penalty weight
    "VREx": (1.0, 10.0, 100.0),
    "GroupDRO": (0.01, 0.1, 1.0),       # group step size
    "CORAL": (0.1, 1.0, 10.0),
    "MMD": (0.1, 1.0, 10.0),
    "DANN": (0.03, 0.1, 0.3),           # gradient-reversal weight
}
K_GRID = (16, 32)
EPOCHS, WARMUP, LR, HIDDEN = 200, 50, 1e-2, 32


def train_predict(method: str, hp: float, k: int, x_tr, y_tr, d_tr, x_te, seed: int) -> np.ndarray:
    """Train one method on (x_tr, y_tr) with domains d_tr; return class probabilities of x_te."""
    import torch

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    def coral(a, b):
        ma, mb = a.mean(0), b.mean(0)
        ca = (a - ma).T @ (a - ma) / max(len(a) - 1, 1)
        cb = (b - mb).T @ (b - mb) / max(len(b) - 1, 1)
        return (ma - mb).pow(2).mean() + (ca - cb).pow(2).mean()

    def mmd(a, b, gammas=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        def kern(u, v):
            return sum(torch.exp(-g * torch.cdist(u, v).pow(2)) for g in gammas)
        return kern(a, a).mean() + kern(b, b).mean() - 2 * kern(a, b).mean()

    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(x_tr)
    pca = PCA(n_components=min(k, x_tr.shape[1], len(x_tr) - 1), svd_solver="full", random_state=seed)
    z_tr = pca.fit_transform(scaler.transform(x_tr))
    z_te = pca.transform(scaler.transform(x_te))
    xt, yt = torch.tensor(z_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    domains = np.unique(d_tr)
    dom_idx = [torch.tensor(np.flatnonzero(d_tr == g)) for g in domains]
    counts = np.bincount(y_tr, minlength=4).astype(float)
    weight = torch.tensor(len(y_tr) / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    feat = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], HIDDEN), torch.nn.ReLU())
    clf = torch.nn.Linear(HIDDEN, 4)
    params = list(feat.parameters()) + list(clf.parameters())
    disc = None
    if method == "DANN":
        disc = torch.nn.Sequential(torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
                                   torch.nn.Linear(HIDDEN, len(domains)))
        params += list(disc.parameters())
        d_t = torch.tensor(np.searchsorted(domains, d_tr), dtype=torch.long)
    opt = torch.optim.Adam(params, lr=LR, weight_decay=hp if method == "ERM" else 1e-4)
    q = torch.ones(len(domains)) / len(domains)
    ce = torch.nn.functional.cross_entropy
    for epoch in range(EPOCHS):
        h = feat(xt)
        logits = clf(h)
        risks = torch.stack([ce(logits[i], yt[i], weight=weight) for i in dom_idx])
        if method == "ERM":
            loss = ce(logits, yt, weight=weight)
        elif method == "IRM":
            pen = 0.0
            for i in dom_idx:
                s = torch.tensor(1.0, requires_grad=True)
                g = torch.autograd.grad(ce(logits[i] * s, yt[i], weight=weight), [s], create_graph=True)[0]
                pen = pen + g.pow(2)
            lam = hp if epoch >= WARMUP else 1.0
            loss = risks.mean() + lam * pen / len(dom_idx)
            if lam > 1:
                loss = loss / lam
        elif method == "VREx":
            lam = hp if epoch >= WARMUP else 1.0
            loss = risks.mean() + lam * risks.var(unbiased=False)
            if lam > 1:
                loss = loss / lam
        elif method == "GroupDRO":
            q = q * torch.exp(hp * risks.detach())
            q = q / q.sum()
            loss = (q * risks).sum()
        elif method in ("CORAL", "MMD"):
            penalty = coral if method == "CORAL" else mmd
            pen = sum(penalty(h[a], h[b]) for ai, a in enumerate(dom_idx) for b in dom_idx[ai + 1:])
            loss = risks.mean() + hp * pen / (len(dom_idx) * (len(dom_idx) - 1) / 2)
        elif method == "DANN":
            loss = risks.mean() + ce(disc(GradReverse.apply(h, hp)), d_t)
        else:
            raise KeyError(method)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(clf(feat(torch.tensor(z_te, dtype=torch.float32))), 1).numpy()


def source_score(method, hp, k, x, y, product, source) -> float:
    """Leave-one-product-out Macro-F1 of one setting on the source products (seed 0)."""
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    pred = np.full(len(y), -1)
    for g in sorted(set(product[source].tolist())):
        tr, te = source[product[source] != g], source[product[source] == g]
        pred[te] = train_predict(method, hp, k, x[tr], y[tr], product[tr], x[te], seed=0).argmax(1)
    return scores(y[source], pred[source])["macro_f1"]


def target_predict(method, hp, k, x, y, product, source, target) -> np.ndarray:
    """Target predictions from the average probabilities of seeds 0, 1, 2."""
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    p = np.mean([train_predict(method, hp, k, x[source], y[source], product[source], x[target], seed=s)
                 for s in (0, 1, 2)], 0)
    return p.argmax(1)


def run_baseline(method: str, x, y, product, n_jobs: int = -1) -> tuple[np.ndarray, dict]:
    """Whole-product holdout of one DG method, with source-only selection of (k, penalty) per fold."""
    from joblib import Parallel, delayed

    y, product = np.asarray(y, dtype=np.int64), np.asarray(product)
    folds = sorted(set(product.tolist()))
    settings = [(hp, k) for k in K_GRID for hp in METHODS[method]]
    jobs = [(o, s) for o in range(len(folds)) for s in settings]
    vals = Parallel(n_jobs=n_jobs)(delayed(source_score)(method, hp, k, x, y, product,
                                                         np.flatnonzero(product != folds[o])) for o, (hp, k) in jobs)
    score = dict(zip(jobs, vals))
    chosen = {o: sorted(settings, key=lambda s: (-score[(o, s)], settings.index(s)))[0] for o in range(len(folds))}
    preds = Parallel(n_jobs=n_jobs)(delayed(target_predict)(method, *chosen[o], x, y, product,
                                                            np.flatnonzero(product != folds[o]),
                                                            np.flatnonzero(product == folds[o]))
                                    for o in range(len(folds)))
    out = np.full(len(y), -1, dtype=np.int64)
    for o, p in enumerate(preds):
        out[product == folds[o]] = p
    return out, {folds[o]: chosen[o] for o in range(len(folds))}
