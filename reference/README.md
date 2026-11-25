# Reference classifier of VET (paper Section IV-B)

VET keeps a source-trained reference classifier as its default answer (Step 1). In the paper it is a hierarchical classifier:

- **Four branches vote over five classes** (straight cruise, curved cruise, hover, ascend, descend):
  - *domain-invariant*: the 192 of 4948 recording features whose class contrast repeats across source products, with the repeatability penalty chosen on held-out source products for every fit, the rank-4 subspace of the source-product means removed, and a radial-basis SVM with soft-margin penalty 10;
  - *condition prototype*: the nearest source condition mean of 36 gain-robust descriptors;
  - *spectral*: an SVM on 454 spectral-activity summaries (raw or standardized within the recording), with the summaries' form, feature selection, removal, kernel, and penalty chosen on held-out source products;
  - *MultiRocket–Hydra*: 54 robustly scaled spectral-activity tracks, 1250 MultiRocket and 8 × 64 Hydra kernels, and a cross-validated ridge classifier.
- **A horizontal expert**, a ridge classifier on slow-time vibration features, separates straight from curved cruise where a periodic-observability check passes; its features, its penalty, and whether it acts are chosen on held-out source products.
- **The domain-invariant branch vetoes** a cruise label when it reports another state, and the condition-prototype branch overrides with Hover where held-out source products favor it.
- Mapping both cruise classes to Horizontal gives the judged state h; the five-class posterior of the domain-invariant branch gives the reference posterior d (eq. (4)).

For VET's held-out source predictions, every selection inside the reference for source product g in fold t is made on the six products left after excluding t and g (strict double exclusion).

## Code

`hierarchical_reference/` holds the authors' research code for this classifier. It keeps only the definitions the pipeline uses, copied unchanged except for imports, the dataset-path default (the environment variable `DRFF_R2_DATASET1`), and the explicit double precision of three functions (see Numerical precision); development notes in comments and docstrings were removed. A comparison of syntax trees (docstrings ignored) confirms that the code was not otherwise changed: of 373 definitions, 365 are identical to the research code, five carry the dataset-path default, and three the precision.

| Module | Role |
|---|---|
| `dataset` | Dataset 1 inventory and the five recorded classes |
| `build_event_tokens`, `event_detector` | burst detection on every recording |
| `build_slow_time_energy`, `slow_time_motion` | slow-time energy around the detected bursts |
| `slow_time_features` | slow-time motion features of the horizontal expert |
| `periodic_observability` | periodic-observability statistics and the gate of the horizontal expert |
| `build_micro_iq`, `micro_iq_extract`, `active_window_scan` | micro-IQ windows, occupancy scan, and probes |
| `scan_stft_features` | scan and STFT recording features (the 4948-d view of the domain-invariant branch) |
| `build_dense_spectrogram` | dense spectrogram scalars of the condition descriptors |
| `build_spectral_activity`, `spectral_activity_features` | spectral-activity tracks and their 454 summaries |
| `branch_domain_invariant` | domain-invariant branch with its nested selection |
| `branch_condition_prototype`, `stacking` | condition-prototype branch and its probabilities |
| `spectral_candidates` | spectral branch candidates and nested selection |
| `branch_multirocket_hydra` | MultiRocket–Hydra branch |
| `router`, `dualtime` | branch evidence for every fit set, the plurality vote, five-class metrics |
| `horizontal_expert` | horizontal expert and its source-only switches |
| `topology_veto` | domain-invariant veto and Hover override |
| `double_exclusion` | held-out source predictions with strict double exclusion |

## Running

```bash
pip install -r requirements.txt
export DRFF_R2_DATASET1=/path/to/DRFF-R2/dataset1-single_drone_states
python reference/run_reference.py features --dataset-root $DRFF_R2_DATASET1 --work work/
python reference/run_reference.py model --work work/ --out reference_outputs.npz
python scripts/run_vet.py views.npz --reference-outputs reference_outputs.npz
```

`features` is resumable and reads the 168 recordings (88 GB) in several passes; it took about an hour with a GPU, on which every check below ran. `model` fits about 120 branch sets and took 10 minutes on a 24-core desktop CPU (the first run also compiles aeon's kernels); its held-out source step runs with single-threaded numerical libraries, as in the paper's run.

## Numerical precision

The paper's micro-IQ and STFT caches were built under NumPy 1, which computes FFTs and float32 quantiles in double precision; NumPy 2, pinned here, computes both in single precision for single-precision input. Three functions therefore state double precision explicitly: the scan scores (`_scan_scores`), the probe descriptors (`_probe_feature`), and the STFT quantiles (`build_stft_features`). Without this, scan scores move by up to 1.8e-7, near-tied scores change places, five of the 168 recordings select one or two other windows of 29, and the scan features of 40 recordings move by more than 1e-4, because they divide by the spread of the recording's scores, which is small where the scores saturate. The STFT features depend on the device: the paper's run used a GPU, and a CPU moves them by up to 2.2e-5.

## Faithfulness

Run on the raw recordings with the pinned versions and a GPU, `features` reproduces the paper's caches bit for bit: event tokens, slow-time energy, micro-IQ windows with their scan scores and probe descriptors, dense spectrogram, spectral-activity tracks and their 454 summaries, slow-time motion features, periodic-observability statistics and gate, and the scan part of the 4948-d view. In the STFT part of that view, one of 256,032 values differs, by 7.5e-9.

`verify_reference.py` compares a fresh run of `model` with the arrays behind the paper (not distributed here). Run on the paper's feature caches, and again on the caches that `features` rebuilt from the raw recordings, `model` gives:

- **Outer folds:** the plurality vote, the horizontal expert's reports (target and held-out source), the domain-invariant reports and posteriors (largest difference 0.0), and the final judgments are identical to the paper's; the reference reports 108 of 168 recordings correctly, as in the paper.
- **Held-out source products:** for all 56 pairs of target and held-out source product, the domain-invariant, condition-prototype, horizontal-expert, and final reports and the domain-invariant posteriors are identical to the arrays that VET read.
- **VET on the fresh outputs:** `scripts/run_vet.py` with these reference outputs reports all 168 recordings as in the paper (118 correct), also with the views that `scripts/extract_views.py` extracts from the raw recordings.
