# Distribution fidelity beyond the 100-mode coverage gate

The retained 20,000-draw live clouds from the selected configuration pass the
original coverage gate on all three problems. They do **not** yet match the
target mixture closely. In particular, square-grid mode means have a root
mean squared displacement of 0.664 target standard deviations, even though
every mode is covered. All three clouds have conditional component variance
15–20% below the target. The [machine-readable audit](accuracy-diagnostics.json)
contains every measured value and the fixed oracle calibration.

| Cloud | Mass TV ↓ | Center RMS / σ ↓ | Conditional covariance trace bias ↓ | Conditional radial KS ↓ | Original gate | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Square grid | 0.08345 | 0.664 | −0.179 | 0.0258 | Pass | Fail |
| Rotated grid | 0.05455 | 0.175 | −0.152 | 0.0634 | Pass | Fail |
| Staggered grid | 0.07500 | 0.240 | −0.196 | 0.0740 | Pass | Fail |
| Target-sampler oracle, 20k p95 | 0.0324 | 0.104 | 0.0109 absolute | 0.00895 | Pass | Pass |
| **Fixed accuracy limit** | **0.060** | **0.200** | **0.100 absolute** | **0.040** | **Required** | **All four required** |

The oracle row is the 95th percentile of 16 independent 20,000-draw samples
from `sample_real`, using a fixed evaluator seed. These draws calibrate the
finite-sample noise floor; they are not training seed experiments. The same
stream is used for each geometry so the oracle comparison isolates the
geometry. The accuracy limits were fixed before the accuracy configuration
search, and the [original frozen gate](../../benchmarks/toy100/metrics.py)
remains mandatory.

The four measurements answer distinct questions. Mode mass TV measures how
evenly the generated draws populate the 100 nearest-center cells. Center RMS
measures the means of the in-radius points, in units of target σ=0.03.
Conditional covariance trace bias compares their average variance with the
analytic variance of a 2D Gaussian truncated at radius 3σ (0.94945σ² per
axis). The radial KS statistic compares the pooled in-radius distances with
that truncated Gaussian's radial CDF. The audit also reports the absolute
difference between generated and target within-radius probabilities; the
target probability is 98.8891%, so extra in-radius mass is not treated as an
improvement by itself. A narrower component raises the radial and covariance
errors. Synthetic shifted, narrow, ring-shaped, and unbalanced clouds all
pass the original gate yet fail the corresponding new check in
[`tests/test_toy100_accuracy.py`](../../tests/test_toy100_accuracy.py).

For ranking eligible configurations, `accuracy_score` is the arithmetic mean
of each of the four errors divided by its fixed limit. Lower is better; a
configuration must first pass the original gate and every individual accuracy
limit. This score is a transparent search objective, not a proof of Gaussian
components or robustness across training seeds. Final comparison should use
independent held-out live draws as well as sustained terminal checkpoints.

Reproduce this audit from the saved clouds:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.toy100.accuracy \
  reports/toy100/recommended --oracle-repetitions 16 \
  --output reports/toy100/accuracy-diagnostics.json
```

## Independent hosted-run audit

The [last original-coverage CI run](https://github.com/255BITS/ParticleGAN/actions/runs/35896027038)
passed all three original coverage gates at commit `2b810a9`. Unlike the
earlier archived samples, this fresh run also retained all five terminal
20,000-draw clouds and separate 100,000-draw holdouts. Reopening that downloaded
evidence with `python -m benchmarks.toy100 accuracy --output DIR` produces a
valid **0/3 accuracy PASS** result, with no missing-evidence failures.

| Problem | Passing final accuracy checks / 5 | 100k mass TV | Center RMS / σ | Covariance trace bias | Radial KS |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid100 | 0 | .0842 | .6561 | −.1802 | .0222 |
| rotated100 | 1 | .0474 | .1520 | −.1558 | .0638 |
| staggered100 | 0 | .0658 | .2170 | −.1938 | .0759 |

These holdouts confirm that counting all modes does not establish accurate
component widths, centers, or mass. The current pull-request workflow trains
and grades all 22 cases under `configs/toy100/shared_candidate.json`; its
three 100-mode rows now require both coverage and this accuracy protocol.
The current shared candidate is failing, and the PR remains a draft.
