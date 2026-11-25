# VET: Verified Evidence Transfer for Radio-Frequency Flight States of Unseen Drones

This repository holds the implementation of the paper *VET: Verified Evidence Transfer for Radio-Frequency Flight States of Unseen Drones* by Xinbin Zhang, Tiantian Tang, and Guan Gui (Nanjing University of Posts and Telecommunications), under review.

A passive counter-drone monitor reads whether a drone hovers, climbs, descends, or passes by from the drone's own radio link. Deployed, it keeps meeting drone models that supplied no training data. Added evidence about the flight then repairs some of its errors on such a model but damages some correct reports. VET keeps the source-trained classifier as the default answer and corrects a report only in directions that held-out known models have verified.

**The rule.** Hold out each known drone model in turn, as if it were new. Count how often each ordered change of state (for example Hover → Horizontal) repairs an error and how often it damages a correct report. Apply to a new model only the directions that repair at least twice, more often than they damage, and in at least two thirds of their changes. No label of the new model is needed.

## What is included

| Module | Paper | Content |
|---|---|---|
| `vet/views.py` | Sec. IV-C | envelope view (454-d): four log-power traces of 16384-sample frames, standardized within the recording, pooled over five time scales, eight temporal bins, trends, bursts, and 16 period bands; STFT view (1524-d) of 64 uniform windows |
| `vet/heads.py` | Sec. IV-C, eq. (5) | the three conditional heads (hover s, vertical v, ascent u), their fit sets F_j, and the paper's seed schedule |
| `vet/candidate.py` | eqs. (4), (6)–(8) | reference posterior d, chained state posterior r, anchored reference posterior b, candidate posterior z |
| `vet/direction_test.py` | eqs. (3), (9)–(11) | **direction test**: repairs, damages, and neutral changes per ordered change of state, admission (R ≥ 2, R > D, R ≥ 2/3 · \|C\|), decision |
| `vet/configuration.py` | eqs. (12)–(13) | **source-only configuration**: eligibility and lexicographic selection of θ = (λ, η) |
| `vet/model.py` | Alg. 1, eqs. (1)–(2) | whole-product holdout with held-out source predictions M_t; `fit_monitor` for deployment |
| `vet/gate.py` | Sec. V-B | confidence-gated correction, the published-rule baseline |
| `vet/analysis.py` | Sec. V-F | held-out source analysis of the correction mechanism (Fig. 8(a)–(b)) and the pooled threshold sweep |
| `vet/baselines.py` | Sec. V-B, Table III | ERM, IRM, V-REx, Group DRO, CORAL, MMD, DANN retrained on the same views (PyTorch) |
| `vet/metrics.py` | Secs. III-D, V-G | accuracy, balanced accuracy, Macro-F1, paired outcomes, exact McNemar, Wilson, sign-flip, product bootstrap |
| `vet/data.py` | Sec. V-A, Table I | DRFF-R2 Dataset 1 inventory: products, conditions c1–c21, four-state map |
| `reference/` | Sec. IV-B | **the paper's hierarchical reference classifier** (Step 1 of VET): feature builders, four branches, plurality vote, horizontal expert, veto and Hover override, strict double exclusion; see `reference/README.md` |
| `vet/reference.py` | — | a simple stand-in reference with the interface VET needs, for use without `reference/` |

## Installation

```bash
git clone https://github.com/Kilimetarlop/vet-drone-flight-states.git
cd vet-drone-flight-states
pip install -r requirements.txt
```

Python 3.10 or later. The pinned versions are those with which every check below was run; PyTorch is needed for the STFT view, the reference features, and the DG baselines, and aeon for the reference classifier.

## Quick start

```bash
python examples/demo_synthetic.py   # VET end to end on a synthetic pool (no data needed)
python -m pytest tests              # checks of the equations and the protocol
```

The demo builds eight synthetic products whose reference classifier reports slow transit as hover. VET admits Hover → Horizontal in every fold and repairs most of these reports. Its numbers describe the synthetic pool only.

## Typical use on your own data

VET wraps any source-trained classifier. Give it a callable `reference(train, test) -> (h, d)` that fits on the recordings `train` and returns, for the recordings `test`, the judged state `h` (0 Horizontal, 1 Hover, 2 Ascend, 3 Descend) and a four-state posterior `d`:

```python
from vet.model import run_whole_product_holdout, fit_monitor

views = {"envelope": X_env, "stft": X_stft}            # (n, 454), (n, 1524) from vet.views, or your own views
result = run_whole_product_holdout(y, product, views, reference)    # evaluation: one fold per product
for fold in result.folds:
    print(fold.target, fold.selection.lam, fold.selection.eta, fold.selection.admitted)

monitor = fit_monitor(y, product, views, reference)    # deployment: all known products as sources
reports = monitor.report(views_new, h_new, d_new)      # h_new, d_new from the reference fitted on all products
```

Everything VET selects (the configuration, the admitted directions) is read from held-out predictions of the known products; no label of the new product enters.

## Running the pipeline on DRFF-R2

```bash
export DRFF_R2_DATASET1=/path/to/DRFF-R2/dataset1-single_drone_states
python scripts/extract_views.py $DRFF_R2_DATASET1 views.npz                                   # VET's two views
python reference/run_reference.py features --dataset-root $DRFF_R2_DATASET1 --work work/      # reference features
python reference/run_reference.py model --work work/ --out reference_outputs.npz              # reference outputs
python scripts/run_vet.py views.npz --reference-outputs reference_outputs.npz                 # VET (Algorithm 1)
```

Each extraction step reads Dataset 1 (about 100 GB) once; `model` takes about an hour on a desktop CPU. `run_vet.py` documents the format of the reference outputs, so any other reference classifier can take Step 1. Without `--reference-outputs`, it fits the stand-in of `vet/reference.py` inside every split. That reference is weak on unseen products (78 of 168 correct), its errors do not recur by direction, and VET admits no change on top of it: VET's gain comes from verifying the recurring errors of a strong reference.

## Faithfulness to the paper

`scripts/verify_paper.py` reruns every step of VET with this package on the arrays that produced the paper's results (per-recording views, the reference classifier's outputs on target and held-out source products, and each method's reports). These arrays come from the authors' pipeline and are not distributed here. The checks:

- **Heads.** Head scores refitted from the views match the saved scores within 3.0e-8, on all eight target products and all 56 held-out source products.
- **VET.** All 168 reports are identical (118 correct; accuracy 70.24%, balanced accuracy 59.64%, Macro-F1 60.71%), and so are each fold's configuration and admitted directions: Hover → Horizontal in eight folds, Descend → Horizontal in seven.
- **Gate.** All 168 confidence-gated reports are identical (110 correct), with the same per-fold thresholds.
- **Mechanism.** On held-out source products, the untested blend proposes 176 changes (94 repairs, 49 damages, 33 neutral); the direction test admits 49, 10, and 9, and the gate 30, 7, and 8. Confidence ranks a repair above a damage with an AUC of 0.596. A pooled threshold that admits 49 repairs admits at least 16 damages.
- **Sensitivity.** The twelve admission-threshold settings and four fixed configurations give 114 to 118 correct reports, as in Table VI.
- **Statistics.** Wilson interval 62.93%–76.64%, McNemar p = 0.021, sign-flip p = 0.0625, and product-bootstrap intervals of ΔAccuracy [+1.79, +9.52], ΔMacro-F1 [−0.46, +6.81], and ΔBA [−3.39, +3.39] points.
- **Baselines.** Retrained with `--with-baselines`, all seven DG methods give the paper's reports for all 168 recordings: ERM 83, IRM 82, V-REx 84, Group DRO 87, CORAL 91, MMD 87, and DANN 80 correct.
- **Reference classifier.** Rerun with `reference/run_reference.py model` on the paper's feature caches, it reproduces every judgment and posterior of the paper's reference on the target and the held-out source products, and VET on these outputs reproduces all 168 reports; see `reference/README.md`.
- **Views.** Re-extracted from four raw recordings of different products and states (`--raw-root`), the envelope view equals the saved view exactly and the STFT view agrees within 8.6e-6 (the windows and the STFT channels are stored as float16).

## Data

DRFF-R2 (Zheng et al., *Scientific Data*, 2026, doi:10.1038/s41597-026-08490-8) is distributed by its providers under their own terms. This repository does not redistribute it or any array derived from it. `vet/data.py` expects the 208 Dataset 1 files (`<product>_<state>_c<k>_u1_d2.mat`) in one folder.

## Not included

- **The arrays behind `scripts/verify_paper.py` and `reference/verify_reference.py`.** They contain features derived from DRFF-R2 (see Data); the scripts document what they compare, and their results are listed above.

## Citation

```bibtex
@article{zhang2026vet,
  author  = {Xinbin Zhang and Tiantian Tang and Guan Gui},
  title   = {{VET}: Verified Evidence Transfer for Radio-Frequency Flight States of Unseen Drones},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
