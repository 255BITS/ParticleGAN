# Hydraulic round two: travel and local deformation

**The combined successor still fails the complete Gaussian and native gates.**
Both arms pass2/4 matched executed tasks. Native precision improves over the
matched winner (.50133 versus.24072), but the .55 forecast is missed and local
width remains excessive. Stop this exact combined revision as a global repair.

One successor adds **network-only antithetic finite-response regularization**
to hydraulic-v1. The primary matched control is the exact BCAP winner. Forge refuses hydraulic-v1
as an admission control because its two-pole binding is unsupported. This
comparison therefore measures the combined travel/deformation successor, not
the incremental causal effect of the added penalty. Round-one hydraulic-v1
measurements remain contextual; they are not a third measured arm. This bounded diagnostic changes no default
or ordinary qualification and preserves all round-one scientific identities.

## Saved-state motivation and mathematical mechanism

[Saved-state analysis](saved-deformation.json) uses the original round-one native
checkpoints without training or sampling. Swapping their complete latent tables
shows hydraulic G's mean local variance .008855 on its own table and .008822 on
the winner table; the winner G gives .005532 on its own table and .008831 on the
hydraulic table. Thus network shape and prior relocation both matter; hydraulic
G's width is insensitive to this particular table swap. Initial public G's
variance is only .000003906 on initial locations. Freezing its initial Jacobian
would prevent substantial necessary expansion. Endpoint network and prior
displacements partly oppose (normalized rowwise dot -.317988), with centered
energies 33.2904 and .605099. These endpoint swaps are sensitivity diagnostics,
not per-update causal decompositions. The analysis checks original checkpoint
hashes, preserves global RNG, and consumes neither target labels nor centers.

Let c be the consumed prior center and epsilon the **already sampled** training
jitter. Define q = (G(c+epsilon)-G(c-epsilon))/2 and
V = mean_i ||x_i-mean(x)||^2 for the consumed generator-side real batch.
For V>0 the successor trains with

`L_G = L_adversarial + existing_prior_term + |stopgrad(L_adversarial)| * mean(||q||^2) / V`.

The single global deformation coefficient is **1**, with unchanged travel
fraction **1**. Both center and jitter are detached in this additional term:
it changes only network gradients; the prior keeps its existing adversarial
gradient. For a locally linear G, E||q||^2 approximates sigma^2 ||J_z G||_F^2.
This discourages expansion of each noise neighborhood while the trainable
centers can supply population transport. The analogy is mechanical strain
alongside travel. It is a finite-response contraction preference, not an
incompressibility law, target-covariance estimate or bound on all Jacobians.

[Odena et al. (2018)](https://arxiv.org/abs/1802.08768) study generator Jacobian
conditioning and introduce Jacobian Clamping. This experiment penalizes the
energy of existing finite-jitter secants rather than clamping condition number
with newly drawn perturbations. The joint parameter proposal is still passed
through the original exact nonlinear hydraulic output-travel replay. No target
centers, component labels, evaluation metric or gate threshold enters training.

The successor requires a positive-width nonstandardized MoG, zero-momentum
DualNorm, clean deterministic training and a stateless supported generator.
It requires a materialized generator-real batch. Preflight computes finite
spacing and variance **before D steps**. Duplicates use nearest distinct positive
distances; a singleton or constant finite batch has radius0 and a complete
zero G/prior parameter ray. D trains normally and the zero-momentum optimizer
clocks advance once. V=0 contributes zero deformation loss. No epsilon floor
silently invents movement. Hydraulic-v1 keeps its original positive-spacing
refusal contract; positive continuous-law comparison batches share spacing.

## Frozen comparison, forecasts and budget

The ready studies are `hydraulic-deformation-candidate-round2-v1` and
`hydraulic-deformation-control-round2-v1`; campaign
`hydraulic-deformation-round2-v1` has a fresh **14,400-second ceiling** and
6,420 seconds per arm. Both arms use the unchanged
`hydraulic_motion_diagnostic` view: Gaussian smoke and its own-state stability,
grid100, vector_two_broad, plus two_pole genuinely BLOCKED for the candidate and measured on the unchanged
winner fixture. A failed own smoke blocks its stability continuation. No substituted
fixture, shortened schedule, new seed, third arm or automatic successor is allowed.
The control's original parent declaration is an untrained admission reference only.

Prespecified numerical prediction: native final receipt `precision >= .55`;
falsifier: `precision < .48`. Additional explanatory forecasts: served
`abs_cov_trace_bias <= .70`, uncensored saved-center median local variance
ratio below the matched winner control, and vector_two_broad PASS retained.
Strict Gaussian/native sustained gates are unchanged; forecasts grant no PASS.
Loss of the passing vector guardrail rejects this revision as a global repair.

Competing explanations: global real variance may underweight local native
density; contraction may prevent useful expansion or encourage prior
compensation; finite antithetic responses may miss activation crossings in
different directions; strict Gaussian mean/width drift may remain. Native
center metrics retain their within-radius population scope. Only the measured
current source/runtime pair supports causal comparison.

Protocol seed0, public deterministic initialization, architecture, prior widths,
mixture weights, component counts, sampling, seen batches, update allowances and
evaluation cadence remain task-owned and fixed. Frozen vector/Gaussian adapters
reuse one real tensor for D/G each update; that actual law is retained. Constructor,
data, prior, training-noise and evaluation streams remain isolated/checkpointed.
No extra RNG draws are added by antithetic secants or replay. Clean/live results
remain distinct from noisy or EMA cohorts. No finite-atom/image/conditional
transfer is claimed; two-pole remains its explicitly separate fixed fixture.

Raw logs and state dumps stay outside Git. Tail:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/hydraulic/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/hydraulic/queue/events.jsonl
```

The public Queue/drain runner uses one active worker on GPU0, authorized sharing,
`watch=False` and `on_completion=None`. Contended wall time is accounting only.
Readout/publication uses summaries-only compilation to preserve archived grades.

## Completed numerical comparison

[Final metrics](results.json), [source and checkpoint receipts](provenance.json),
[actual-training GIF index](media/index.json), [four exact Gaussian misses](gaussian-misses.json)
and [all-center local-width diagnostic](native-width.json) provide the evidence.
Scoring uses complete numerical gates; no image inspection contributes a grade.

| Unchanged task | Combined successor | Matched exact winner | Main measured result |
| --- | --- | --- | --- |
| two_pole | BLOCKED, no attempt | PASS | Replay extensions unsupported by the unchanged component host |
| gaussian1d_smoke | PASS | PASS | First confirmed875 versus375; endpoint KS.0453718 versus.0718422 |
| gaussian1d_stability | FAIL | FAIL | Stationary69/72 versus2/72; shifted hold23/24 versus0/24; reacquisition PASS versus FAIL |
| grid100 | FAIL | FAIL | 100k holdout precision.50133 versus.24072; terminal accuracy0/5 for both |
| vector_two_broad | PASS | PASS | Passing suffix18/24 versus22/24; full covariance error.277472 versus.385581 |

Declared outcomes are **2 PASS, 2 FAIL, 1 BLOCKED** for the candidate and
**3 PASS, 2 FAIL** for the control. Two-pole is an additional control-only fixed
fixture, not a measured candidate failure or superiority cell. Four paired
conditions pass exact recipe, prior, initialization, host/task and consumed
training-stream checks. Gaussian records actual batch-sequence digests; native
and vector batch parity follows identical laws, seeds, draw/update counts and
checkpointed data-stream states. No unrecorded batch digest is invented.

Both Gaussian continuations restore their own actual update1000 endpoint,
including optimizer histories and streams. The candidate begins from a passing
endpoint; the control endpoint already fails KS. No best state replaces either.
The candidate's deadline reacquisition passes; both frozen pre-shift copies
pass0/48 shifted checks. Renewed learning is observed, while complete retention
still fails. Candidate retained failures exceed only the unchanged CDF KS.05:

| Update / phase | Target KS | Width ratio | Mean error / sigma | Fitted-normal KS from the same saved array |
| --- | ---: | ---: | ---: | ---: |
| 2709 stationary | .05774277 | 1.02576119 | .14285450 | .02425609 |
| 2875 stationary | .05910861 | .94249542 | .12958863 | .01836869 |
| 2959 stationary | .05322723 | .94369847 | .09750800 | .01770926 |
| 5542 shifted hold | .06945881 | 1.06650688 | .13383555 | .02661397 |

The saved4096-draw analyses reproduce every target KS exactly. Individual
moment bounds pass; their allowed offsets can still violate the target CDF.
Fitted-normal KS is descriptive and supplies no replacement gate or regrade.
Final Gaussian KS.0160907 versus.320623 cannot replace the failed hold windows.

Broad-vector retains the full PASS: candidate normalized sliced Wasserstein
.128167, mass TV.083496, HQ.993652, full component covariance error.277472 and
minimum eigen ratio.625570 pass every required terminal bound. Its terminal
suffix is18/24, compared with22/24 for the control; a better endpoint metric
does not establish faster or more consistent acquisition.

For native, precision.50133 remains far below.97. Within-radius center RMS
is.363620 target sigmas versus1.520315; this is the evaluator's censored
population and does not measure all served centers. Mass TV.0690 versus.14718
improves but still exceeds.06. Radial KS.333940 versus.490883 exceeds.04.
Covariance trace bias **+.901833 versus+.371385** worsens and exceeds absolute.1.
All five terminal accuracy checks fail, as do complete coverage/holdout gates.
The oracle passes (precision.98914, center RMS.041738, radial KS.001514).

The separate CPU FP64 diagnostic at all20000 saved centers yields median local
MoG total variance / target total variance **4.456226 versus2.976159**, mean
4.950371 versus3.073333, and fraction above1 of99.565% versus97.560%. Thus the
forecast of lower local width than the matched winner is also missed.
Independent reverse-mode derivatives agree within2.67e-15; checkpoint digests
and global RNG stay unchanged. This linearized metric omits activation crossings,
finite jitter and assignment truncation; it is not served-law qualification.

All7000 native proposals are travel-limited, with no rejected motion and maximum
accepted/radius ratio.99999944. Mean proposed joint RMS.240595 becomes.00840435
against mean radius.0119093. The extra contraction term is active: mean secant
energy.00929374 and real total variance16.4935 give a **ratio of pooled sums
.000563479**; mean extra loss is.000432263. Gaussian corresponding pooled ratio
is.0110095 and mean extra loss.00799513. These are training diagnostics, not
gradient-work ratios. As an inference, normalization by global between-mode
variance may underweight local native width. The observed small loss does not
prove the normalized optimizer ignored it; that needs a direction/work analysis.
Zero-spacing counters are0 on every actual continuous-law task. The repaired
zero-ray behavior has software checks, not finite-atom quality evidence.

Round-one hydraulic-v1 at its original source had native precision.49586,
median local width4.618359 and Gaussian retention71/72. Those archived figures
are context only. The changed current source and absence of a matched v1 arm
prevent attributing their differences to the added secant term. The new winner
control exactly reproduces the archived Gaussian/native endpoint metrics,
which is useful implementation parity but grants no cross-source pass credit.

## Disposition and exact provenance

Stop the exact combined revision; retain the public winner/default and original
qualifications. It preserves the broad-vector guardrail and improves placement
over its matched winner, but converts no failing task into a sustained PASS.
The precision.55, covariance.70 and lower-local-width forecasts are all missed.
Do not fill the capability gap, sweep the coefficient, repeat a seed, continue
training or claim image/conditional/finite-atom transfer. Before another separately
bounded structural hypothesis, inspect deformation-gradient work and distinguish
local training density from global variance. This is a research recommendation,
not authorization for another experiment.

Both ready studies are concluded. The [candidate readout](../../../records/readout-9f6ac354828f6069c1e83b87.json)
retains automatic **INCOMPLETE**, because the declared two-pole cell is unmeasured;
precision.50133 satisfies neither the prediction>=.55 nor falsifier<.48.
The [control readout](../../../records/readout-783482660df5696a6ed50f75.json)
retains automatic **FALSIFIED** on that inherited precision signature. It does
not compare experimentally against its untrained admission parent or change
the winner's original qualifications. Numerical task FAILs remain distinct from
unsupported capability and execution incompleteness.

The exact trained commit is
`38c6cd52796eb1b760ae025eadd8a98c9703c101`, scientific digest
`2ebd7377a68b5db8783006c3cb59ff84b7a625027974bf3387db722a940f8357`.
Publication verifies all1201 scientific file hashes still match; no subsequent
scientific correction is published as measured. Candidate revision
`fa836befe1a9b010280a21f7a487a096cf5206686ca49f91078890499410269e`, request
`924a53edb67811f7ba0d29d1`; winner revision
`607425144b6578b5887c26af2ced7a8637087d929584613f268dcf64b7b5edb8`, request
`76967ee468dc4797a66821a7` bind the recorded attempts and checkpoint hashes.
Python3.12.13, Torch2.14.0, NumPy2.5.2, SciPy1.17.1, RTX A6000, one Torch thread,
seed0 and deterministic algorithms match across arms. The exact winner recipe
is non-saturating/fullDualNorm momentum0, smoothing.001, per-offset convolution,
constant G/E.012, D.018, prior.030; BCAP cap/coefficient1/every update, floors1,
zero additive train noise and no EMA. All task conditions/sampling stay fixed.

Nine workers completed **28,480 new host updates**, with zero scientific retries,
errors or incomplete workers. Paid worker cost is **959.771671 seconds**,
zero live reservation, launched full allowances12,540 versus prespecified maximum
12,840 and ceiling14,400. Saved-state analyses cost5.259026 and2.061551 CPU seconds
separately; neither trains or draws evaluation samples. There is no speed ranking
under authorized sharing. The drain has stopped with no leftover watcher.

Validation: **199 focused software/scorer checks pass**, including exact
checkpoint continuation, unchanged RNG consumption, network-only gradients,
zero-spacing complete updates, pre-mutation invalid-input refusal and retained
scorer oracle/destructive controls. Forge declaration validation and summary-only
memory freshness pass. Nine actual-training GIFs and compact receipts are
published; raw stdout, JSONL, checkpoints and state dumps remain in the local queue.
The parent owns the single current cross-track comparison/leaderboard.

Reproduce from the exact trained source, both ready studies and the task laws:

```sh
PYTHONPATH="$PWD" .venv/bin/python reports/forge/bcap-physics/hydraulic/round2/run.py enqueue --queue-root /path/to/new/queue
PYTHONPATH="$PWD" .venv/bin/python -u reports/forge/bcap-physics/hydraulic/round2/run.py drain --queue-root /path/to/new/queue --gpu 0
```

Recreate publication from these existing certified artifacts without training:

```sh
.venv/bin/python reports/forge/bcap-physics/hydraulic/round2/publish.py --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/hydraulic/queue --media
.venv/bin/python reports/forge/bcap-physics/hydraulic/round2/gaussian_miss_analysis.py
.venv/bin/python reports/forge/bcap-physics/hydraulic/round2/native_width_analysis.py
.venv/bin/python -m experiments.forge compile --summaries-only
```
