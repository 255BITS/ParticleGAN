# DDGAN failure investigation

The saved models have a genuine low-noise distribution mismatch. The problem
also appears at central and freshly drawn contexts, so the failures cannot be
explained solely by the original fixed queries being unusual. High-noise
transition passes conceal substantial errors in the vanilla DDGAN's clean-data
prediction because the public reverse kernel supplies most of their variance.
The saved critic's local output gradients often oppose the needed mean
correction. That observation motivates a critic-tracking diagnostic; it does
not identify a causal optimizer defect or establish a working repair.

Both original Gaussian runs remain **FAIL**, and both deeper bimodal full runs
remain **BLOCKED**. AE-DDGAN still passes its separate paired reconstruction
subgates. No training, continuation, seed trial, gate or tier change was made.
The existing [Forge leaderboard](../technique-inventory.md) remains the single
leaderboard for this goal.

This is an **exploratory follow-up**. The reports, curves and final states were
inspected before implementing this analysis; the protocol is neither a blind
preregistration nor independent validation. The [protocol](protocol.json) and
[compact results](results.json) record its exact scope and provenance. These
diagnostics supply no qualification credit.

## Evidence and method

The original source is
[`bee9f06ed21695e518c1cb702e12a385ab9389e8`](https://github.com/255BITS/ParticleGAN/tree/bee9f06ed21695e518c1cb702e12a385ab9389e8).
Both runs used the same `bispf-optimized-quintic-v1` recipe, protocol seed 0,
public deterministic initialization, the declared learned MoG prior, clean live
serving, CPU and one Torch thread, for all 2,500 updates. Their complete task,
recipe and original numerical failures remain in the
[original DDGAN](../api-model-toys/ddgan.md) and
[AE-DDGAN](../api-model-toys/ae-ddgan.md) readouts and the
[original publication](../../toy_audit/api_contract/model_modes/publication.json).

The runnable [diagnostic](diagnose.py) verifies all 1,252 frozen manifest files,
allowing only the three already published exact reporting-only replacements.
It then imports the frozen scientific code and checks the committed hashes of
each original receipt, adapter receipt, checkpoint, observation array and GIF
before loading the trusted local checkpoint. Current develop's changed trainer
code is not used to interpret the saved run. It restores detached forward models
and original prior-reference buffers, without constructing an optimizer or
loading/consuming a training stream. Loaded models, gradients, modes, caller
Torch RNG and original artifact hashes remain unchanged.

New diagnostic sampling uses isolated streams with seed 0: 4,096 draws per
panel and eight fresh contexts per timestep drawn from the exact Gaussian
forward marginal. The target, network and checkpoint stay fixed. These are
additional endpoint measurements, with their own evaluation streams and sample
count; they do not replace or regrade the original 2,048-draw, 24-observation
training protocol. Reported context medians are ordinary medians of eight
point estimates, without a robustness or confidence-interval claim.

## What the numbers establish

For Gaussian clean data, the exact conditional reverse covariance is
`A_t² Var(x0 | xt) + posterior_var_t I`. The fraction due to the learned clean
sample is therefore analytically known:

| Reverse step | Coordinate 1 | Coordinate 2 |
| --- | ---: | ---: |
| 1 | 100% | 100% |
| 2 | 57.40% | 39.78% |
| 3 | 15.95% | 7.82% |
| 4 | 1.05% | 0.47% |

This is a variance decomposition, not a claim that those later steps contain
no learned information. It explains why their distribution gates are less
sensitive to a poor clean predictor. At the vanilla checkpoint's `t4_q1`, the
new clean-posterior measurements give covariance eigenvalues **[0.114, 0.535]**
and projection CDF error **0.271**, while the actual reverse transition gives
eigenvalues **[0.971, 0.980]** and CDF error **0.033**. A destructive control
collapses the clean predictor to the exact posterior mean: the actual step-4
reverse samples still pass the unchanged transition bounds, while step 1
correctly rejects their missing variance. This control uses public
`DDGAN.reverse`; it is separate from the learned baseline.

The original `t1_q0` query has forward-marginal Mahalanobis distance **2.434**
(5.18% Gaussian radial tail), while `t1_q1` is central at **0.233** (97.33%
radial tail). The central query still has fresh diagnostic CDF errors **0.214**
for DDGAN and **0.256** for AE-DDGAN. Its normalized mean errors are **0.581**
and **0.480**, respectively. This rules out a purely unusual-query explanation.

| New endpoint measurement at step 1 | DDGAN | AE-DDGAN |
| --- | ---: | ---: |
| Median transition CDF error over 8 marginal contexts | 0.265 | 0.301 |
| Maximum transition CDF error | 0.416 | 0.450 |
| Median normalized mean error | 0.747 | 0.641 |
| Maximum normalized mean error | 1.001 | 1.253 |

The fresh contexts confirm that the mismatch extends beyond the two frozen
queries. They do not certify the full continuous conditional law.

At the vanilla endpoint, descending each example's actual non-saturating
generator loss in **output space** opposes translating generated samples
toward the analytic conditional mean on **7/8 fixed panels**, including both
step-1 panels. For example, the step-1 central-panel alignment is **-0.00431**;
positive would assist that mean correction. It is also negative on all eight
fresh step-1 contexts. In AE-DDGAN, alignment is negative on **4/8 fixed
panels**, including the central step-1 panel (**-0.000604**), and on 6/8 fresh
step-1 contexts. Small later-step signs are point estimates, not significance
tests.

These gradients are negative **per-example** loss derivatives with respect to
`x_prev`, before the batch-mean `1/n` factor. They are not parameter updates:
the generator Jacobian, shared parameters, reconstruction/prior objectives and
DualNorm transformation can change the actual update's effect. Endpoint
gradients alone cannot prove critic lag, optimizer overshoot or a temporal
cause. Positive/negative linear-critic controls verify the alignment sign.

The saved final five observations already show an unsettled terminal marginal:
DDGAN final CDF error spans **0.202–0.474**, and AE-DDGAN spans **0.147–0.286**
(gate <=0.12). AE-DDGAN's paired NMSE remains **0.0244–0.0441**, worst-axis
NRMSE **0.170–0.213**, and shuffled-code gain **45.17–79.13**, all within their
separate bounds. Correct input pairing therefore coexists with incorrect
independent generation in this exact cohort.

## Next bounded diagnostic

A useful next test would freeze each saved G/prior/encoder and fit only its
existing critic on fixed public forward/reverse transition pairs for at most
256 critic updates. Freeze the batch sequence, held-out contexts and one global
critic recipe before spending, measure held-out logistic loss and the same
output-gradient alignments, and use a separate diagnostic cohort with its own
numerical gate and actual-training GIF. If the fixed critic can learn a
corrective field against the stationary generator, that would support capacity
being sufficient for that measured task and motivate a controlled tracking
experiment. Failure within that budget would not prove a capacity limit.

This proposal is not executed or selected as a repair. A later trainer
comparison must use one global recipe on both unchanged Gaussian tasks, retain
their full budgets/cadence and original numerical gates, and preserve the
current failures. More updates, per-task tuning or weaker gates are not
supported by this saved-state analysis.

## Reproduction and checks

Prepare the frozen checkout using the
[original source instructions](../api-model-toys/README.md). A raw detached
checkout at `bee9f06ed` also works; the exact reporting overlay is optional.
Use an exclusive output filename. No model update or new GIF is produced:

```sh
PYTHONPATH=. python reports/forge/ddgan-failure-investigation/diagnose.py \
  --source-root /path/to/frozen-bee9-checkout \
  --artifacts /nvme/dev/particlegan/artifacts/api-model-toys/research-v1 \
  --output /path/to/new-diagnostic-results.json > /path/to/diagnostic.log 2>&1
tail -f /path/to/diagnostic.log
```

The completed local log is
`/nvme/dev/particlegan/artifacts/ddgan-failure-diagnosis/diagnostic-v3.log`.
Earlier loader-development attempts and report clarifications remain in that
external directory; they made no training updates. The compact final results
are committed; raw checkpoints, traces and execution logs stay outside Git.

The original actual-training GIFs remain unchanged:
[DDGAN](../../toy_audit/api_contract/model_modes/media/api-model-ddgan-gaussian-chain.gif)
and [AE-DDGAN](../../toy_audit/api_contract/model_modes/media/api-model-ae-ddgan-gaussian-chain.gif).
The eight focused tests cover source/artifact mismatch refusal, independent
posterior variance algebra, noise masking under collapsed predictions,
alignment sign controls, unsupported loss refusal, state/RNG purity and exact
published evidence identity.
