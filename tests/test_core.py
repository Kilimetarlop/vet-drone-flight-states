"""Fast checks of the equations and the protocol (python -m pytest tests)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vet import metrics  # noqa: E402
from vet.candidate import anchor, blend, candidate_posterior, chained_posterior, reference_posterior  # noqa: E402
from vet.configuration import select_configuration  # noqa: E402
from vet.direction_test import admitted_directions, decide, direction_counts, outcomes  # noqa: E402
from vet.model import make_precomputed_reference, run_whole_product_holdout  # noqa: E402
from vet.views import FRAMES, envelope_frames, envelope_view, stft_view  # noqa: E402


def test_chained_posterior_is_eq6():
    s, v, u = 0.2, 0.7, 0.4
    r = chained_posterior(np.array([[s, v, u]]))[0]
    expected = [(1 - s) * (1 - v), s, (1 - s) * v * u, (1 - s) * v * (1 - u)]
    assert np.allclose(r, expected) and math.isclose(r.sum(), 1.0)


def test_reference_posterior_is_eq4():
    q = np.array([[0.1, 0.2, 0.3, 0.25, 0.15]])
    assert np.allclose(reference_posterior(q), [[0.3, 0.3, 0.25, 0.15]])


def test_anchor_multiplies_the_odds_of_the_judged_state():
    d = np.array([[0.5, 0.3, 0.1, 0.1]])
    b = anchor(d, np.array([1]), math.log(3.0))[0]
    assert math.isclose(b[1] / b[0], 3.0 * 0.3 / 0.5, rel_tol=1e-9)
    assert math.isclose(b[2] / b[0], 0.1 / 0.5, rel_tol=1e-9)


def test_blend_is_eq8():
    r, b = np.array([[0.7, 0.1, 0.1, 0.1]]), np.array([[0.1, 0.6, 0.2, 0.1]])
    assert np.allclose(blend(r, b, 0.4), 0.4 * r + 0.6 * b)


def test_candidate_follows_the_chain_when_the_reference_is_unsure():
    d = np.array([[0.30, 0.34, 0.18, 0.18]])                    # judged Hover, barely
    z = candidate_posterior(d, np.array([1]), np.array([[0.05, 0.1, 0.5]]), 2 / 5, math.log(3.0))
    assert z.argmax(1)[0] == 0                                  # candidate Horizontal


def test_direction_test_is_eq10():
    # Hov->Hor: 2 repairs, 1 damage (share 2/3: admitted); Des->Hor: 2 repairs, 0 damages, 2 neutral (share 1/2: no);
    # Asc->Hor: 1 repair (R < 2: no)
    h = np.array([1, 1, 1, 3, 3, 3, 3, 2])
    c = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0, 0, 1, 0, 0, 1, 2, 0])
    a = admitted_directions(h, c, y)
    assert a[1, 0] and not a[3, 0] and not a[2, 0]
    cnt = direction_counts(h, c, y)
    assert (cnt["repairs"][1, 0], cnt["damages"][1, 0], cnt["neutral"][3, 0]) == (2, 1, 2)
    assert list(outcomes(h, c, y)[:3]) == ["repair", "repair", "damage"]
    assert list(decide(h, c, a)) == [0, 0, 0, 3, 3, 3, 3, 2]


def test_selection_breaks_ties_toward_small_lambda_and_large_eta():
    y = np.repeat([0, 1, 2, 3], 30)
    product = np.tile(np.arange(6), 20).astype(str)
    h = y.copy()
    h[:12] = 1                                                  # recurring Horizontal -> Hover errors
    d = np.full((120, 4), 0.25)                                 # a flat reference posterior
    s = np.column_stack((np.where(y == 1, 0.9, 0.01), np.where(y >= 2, 0.9, 0.01), np.where(y == 2, 0.9, 0.1)))
    sel = select_configuration(y, product, d, h, s)             # every configuration repairs the same 12 reports
    assert sel is not None and sel.admitted[1, 0] and sel.admitted.sum() == 1
    assert math.isclose(sel.lam, 1 / 3) and math.isclose(sel.eta, math.log(4.0))


def test_paper_statistics():
    assert math.isclose(metrics.paired(np.array([0] * 16), np.array([1] * 13 + [0] * 3),
                                       np.array([0] * 13 + [1] * 3))["mcnemar_exact_p"], 0.021270751953125)
    lo, hi = metrics.wilson(118, 168)
    assert (round(100 * lo, 2), round(100 * hi, 2)) == (62.93, 76.64)
    assert metrics.sign_flip_p(np.array([-1, 2, 2, 3, 1, 0, 1, 2])) == 0.0625


def test_views_have_the_paper_dimensions():
    rng = np.random.default_rng(1)
    frames = envelope_frames(rng.normal(size=4 * 16384).astype(np.float32),
                             rng.normal(size=4 * 16384).astype(np.float32))
    assert frames.shape == (4, 4)
    trajectory = rng.normal(size=(FRAMES, 4)).astype(np.float32)
    assert envelope_view(trajectory).shape == (454,)
    assert stft_view(rng.normal(size=(64, 4, 32, 96)).astype(np.float16)).shape == (1524,)


def test_stft_view_does_not_depend_on_memory_layout():
    channels = np.random.default_rng(3).normal(size=(64, 4, 32, 96)).astype(np.float16)
    assert np.array_equal(stft_view(channels), stft_view(np.asfortranarray(channels)))


def test_protocol_runs_with_precomputed_reference():
    rng = np.random.default_rng(2)
    products = [f"p{i}" for i in range(4)]
    product = np.repeat(products, 21)
    y = np.tile(np.array([0] * 12 + [1] * 3 + [2] * 3 + [3] * 3), 4)
    views = {"envelope": rng.normal(size=(84, 20)) + y[:, None], "stft": rng.normal(size=(84, 30)) + (y == 2)[:, None]}
    h = y.copy()
    h[rng.random(84) < 0.2] = 1
    d = np.full((84, 4), 0.1)
    d[np.arange(84), h] = 0.7
    ref = make_precomputed_reference(product, products, h, d, np.tile(h, (4, 1)), np.tile(d, (4, 1, 1)))
    res = run_whole_product_holdout(y, product, views, ref)
    assert res.report.shape == (84,) and set(res.report.tolist()) <= {0, 1, 2, 3}
    assert [f.target for f in res.folds] == products
