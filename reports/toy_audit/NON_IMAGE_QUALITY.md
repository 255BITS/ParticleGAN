# Stronger non-image toy definitions

This follow-up implements new evaluators for all **22 poor-rated vector cases**
and the other **eight weak non-image definitions**. Alongside the
[34 poor-rated image improvements](IMAGE_QUALITY_V2.md), that covers every
**64 original entries rated 3/5 or 2/5**. Original ratings and training verdicts
remain the frozen audit; stronger gates have separate versioned results.

## Vector density and comparison fidelity

[The vector receipt](vector-quality-controls.json) evaluates **34 retained
training arms** on the original 24 observed clouds per arm. It adds a maximum
KS distance of **.06 over 32 fixed projections**, using the analytic Gaussian
mixture CDF. Each arm must also pass its original gates and five terminal
checks. The target is scored at each cloud's actual step, including declared
scale schedules. Component labels are not required for overlapping mixtures.

All **22 independent target draws** and **22 constructive finite-particle
witnesses** pass both gates. Each finite witness has the declared 256 atoms,
quantile coverage and target moments, then uses an isolated 4,096-row draw;
it is an evaluator control, not a trained generator. All 22 zero-width
center-only controls fail the new gate. The 109 controls also exercise
mean collapse and missing components. Unit rescaling preserves the new metric.

The overlapping-ring stress exposes why this is useful: its original sole
SW1 bar accepts the exact eight centers with **zero component width**
(SW1 .09376 < .18). The new projected CDF check rejects that discrete law.
It also detects mass and shape errors that a broad average transport bound
can tolerate. This is a finite projection test, not a claim that every possible
two-dimensional discrepancy is identifiable.

**13 historical passes become zero stronger terminal passes across these
34 arms.** This is a stricter, separately calibrated diagnostic result; it
does not relabel old receipts or imply the original algorithms newly changed.
The smallest endpoint shortfall is the overlapping-ring model's KS about
.063 against .06. Other former positives have larger residual errors, up to
about .209. A new gate exposes deficits; it does not repair their optimizers.

The 22 variants represent **15 exact target laws**, with host settings and
individual verdicts still listed. The nine stress variants mostly reuse the
same ring law. They test optimizer/capacity/cadence robustness; they do not
provide nine independent data-family wins.

All **12 poor polygon/scale PR contrasts change both the critic and the
particle initializer**. The initializer is absent from their effective
specs, but the actual compressed arm receipts bind it: published std .5,
control std **4–12**. The critic also changes width/depth, Fourier features
and batch features. The complete control establishes solvability under that
complete host. It does not isolate a kernel-lengthscale remedy. The new
machine-readable factor audit exposes the initializer and retains its exact
artifact hash. A causal claim about one critic component needs a separately
registered matched intervention, not another seed or an implicit recipe port.

Reproduce without training:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.toy_audit.vector_quality \
  --artifacts /ml2/hypergan/toy-audit-artifacts-20261001 \
  --curves /tmp/vector-quality-rescore-curves \
  --output /tmp/vector-quality-controls.json
```

## Remaining weak definitions

[The definition receipt](definition-quality-controls.json) supplies positive
witnesses and discriminating controls for every row below. These controls
perform no training. They establish what a stronger test measures; fresh
trained qualification remains separate.

| Original catalog entry | Implemented stronger contract | Negative control that matters |
|---|---|---|
| `develop-two_pole` | Balanced mass, support and sorted quantile fidelity against the actual 12-atom training grid; travel remains a separate original check. | One pole passes old travel; two centers pass coverage but fail width. A 256-row sampler call is a different finite law and cannot calibrate this host. |
| `source-family-16`, Gaussian quickstart | Mean, covariance eigenvalues, radial CDF and 16 projected CDFs of N((1,1), .04 I). | An isotropic circle has correct moments but the wrong law; a point at the mean also fails. |
| `source-family-15`, five words | All six categorical tokens including padding, confidence, uniform word mass and correctly paired reconstruction. | Correct argmax with nearly uniform probabilities; correct generated marginal with permuted reconstruction; display-equivalent wrong padding. |
| `source-family-11`, sign lander | Held-out paired action error relative to a nonzero neutral witness. | Reflecting both generated state and action preserves the entire joint data law while reversing the action for the actual caller; relative MSE is 4. |
| `source-family-13`, native paired action | Previous-command/action correspondence, independent of state and action marginals. | Reversed previous-command pairs preserve the action marginal and fail correspondence. |
| `source-family-12`, safe-fast lander | Every initial state counts: landings, crashes, timeouts and censored mean landing time. | A few quick successful starts cannot hide timeouts; hover and crash policies both fail. |
| `source-family-10`, ring protocols | Eight equal-weight modes, within-mode covariance/radial law, current target after shift and terminal stability. | Centers without width; frozen pre-shift cloud; transient/best checkpoint pass. |
| `source-family-14`, routed fixture group | Paired residual fidelity and signed useful-code contribution under every judge, with each support/moving/replay protocol kept distinct. | Wrong pairs, harmful code, or one disagreeing judge fail. An eight-update replay check is software evidence, not convergence. |

The paired-action oracles do not expand the task into full physical control.
The five words remain a fixed vocabulary; the Gaussian remains a unimodal
plumbing example. A stronger measurement does not create unseen-word,
large-image or task-transfer evidence. The ring and routed definitions still
need their own trained state, declared phases and exact serving law.

For `two_pole`, the actual source uses twelve real rows and twelve clean
particles. Each pole's target offsets are the six endpoint-inclusive linspace
values from -.05 to .05. Increasing the sampler argument changes that discrete
law; a 256-row control was therefore not a valid oracle for this host. The new
positive control enumerates its exact twelve target atoms, and the sorted
quantile gate avoids applying a continuous-CDF KS formula to discrete jumps.
Its original endpoint mean absolute location is .514372. Any in-support atom
has absolute location at least .949999, so at most **six of twelve** atoms can
be in support. The original travel PASS therefore fails the added support bar
without a new training run. Coincident initialization does **not** prove that
the rows remain identical: original paired RpGAN uses different real logits
per row and produces distinct initial gradients.

The ring oracle describes an adequately resolved Gaussian distribution. It
does not prove feasibility for every historical finite host: the standalone
12-particle `mode_hold` cannot supply nondegenerate 2D covariance in all eight
modes, and the captured continuous probe resolves a twelve-row clean prior
plus output noise. Its exact resources and noisy/clean laws stay separate in
the fresh source report.

The committed vector receipt stores endpoints, suffixes and archive hashes.
Its full 816 checkpoint rescoring records remain outside Git under the
declared `--curves` archive. Fixed-projection positive witnesses demonstrate
attainability for the declared finite clouds; they do not estimate a general
false-rejection probability or establish optimizer reachability.

Reproduce controls:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.toy_audit.definition_quality \
  --output /tmp/definition-quality-controls.json
python -m pytest -q tests/test_toy_definition_quality.py tests/test_toy_vector_quality.py
```

The [failure diagnosis](FAILURE_DIAGNOSIS.md) identifies actual failed bars,
structural counterexamples, sampling-law mismatch and unresolved optimizer
questions under the original evidence. It is not replaced by this new cohort.
