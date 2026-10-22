# BISPF settling repair: apply cooling to finite movement

Status: round one complete; **no repaired gates and no lost control passes**.
The candidate is `bispf-settling-displacement-v1`. The common three-theme study
owns the finite campaign, common-source execution and one comparison leaderboard.
This document adds no qualification, default change or historical regrade.
The [measured readout](settling-results.md) and [compact receipt](settling-results.json)
record all 15 tasks: 4 PASS / 11 FAIL, 14 exact control trajectories at retained
observations and checkpoints, and one changed but still failing Grid100 run.

## Question and prior evidence

Can the selected BISPF recipe retain an already reachable distribution by
making its actual G/prior movement shrink with its existing thermal signal,
while preserving acquisition and reheating after a target shift?

The selected original cohort is `bispf-optimized-quintic-v1`, source
`d9d72d53761c898bce20102e1995fc3d899c1447`, protocol seed 0, public deterministic
initialization and original task-owned learned MoG capacities, widths and laws.
The retained readout is
`/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`,
SHA256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`.
Gaussian stability passes only 6/72 stationary observations and 2/24 shifted
hold observations; overlap passes 10/24 observations, with longest run 3 and
required terminal suffix 5. Original checkpoint/sample tensors are unavailable;
these observations motivate a falsifiable new comparison and do not establish
the cause. No unchanged replay is proposed to reconstruct them.

Read prior failures before choosing this delta:

| Prior study | Result and consequence |
| --- | --- |
| [#398 idle rule](https://github.com/255BITS/ParticleGAN/pull/398) | Skipping G/prior on a paired-gap confidence signal loses ring16; Gaussian and mode hold still fail. No idle signal or threshold is reused here. |
| [#429 relative tether](https://github.com/255BITS/ParticleGAN/pull/429) | Capping the latent tether gradient at .03 of other force loses both mode hold and ring16. No relative-gradient cap is reused here. |
| [#435 harmonic link](https://github.com/255BITS/ParticleGAN/pull/435) | Capture-off retains the mode-hold pass and is bit-identical through update 1050. Thermal capture did not explain that repair; small critic-penalty changes caused trajectory-sensitive outcomes. No mode-hold causal explanation or capture threshold change is claimed here. |

The current optimizer applies smoothed normalized matrix, bias and sampled-row
updates. In a prescribed high-gradient software fixture, multiplying a bias
gradient by .1 retains over 99.9% of its original finite movement; multiplying
the finite movement by .1 realizes the intended ratio. This establishes the
mechanism's numerical distinction, not its training efficacy.

## Explicit global trainer and prior delta

Set only `bispf_thermal_application="displacement"` against the full selected
recipe. The default `"gradient"` keeps the existing implementation.

1. Capture the earned critic observer's thermal scale `t` for the G phase.
2. During G/E/prior forward and backward, temporarily make both thermal readers
   return one: the critic forward hook's record and SBNP's device-local cache.
   Thus the combined declared adversarial plus SBNP objective gradient is
   normalized before thermal damping.
3. Complete the existing optimizer step, including direction blending. Replace
   each optimizer-owned tensor by `old + t * (base - old)`, using `torch.lerp`.
4. Restore the original thermal record and cache exactly, even on an exception.

The prior remains learned with SBNP objective weight .01. Its **gradient cooling
policy and applied update do change**: the usual SBNP moment-only cooling is
removed before normalization, as is its special complete low-dimensional
gradient cooling at `t <= 1e-4`; the final combined finite prior displacement
then receives `t` once. Width, mixture weights, initial locations/capacity,
sampling and sampled-row ownership remain task owned. This is a combined change
in the location of thermal application, not a claim to hold prior force ratios
fixed while changing the applied update.

Critic gradients, penalty, residual observation and fast-attack reheating retain
their original law. No new residual threshold, time schedule, loss term,
per-task setting, idle decision, oracle target information or random draw is
introduced. Learning rates, momentum, smoothing and all other recipe values are
unchanged. At `t=1` the optimizer step is bit exact. Helpers have no persistent
state: the already checkpointed observer, optimizer and named streams suffice
for exact continuation.

## Predictions, falsifiers and bounded decision

Use the parent study's one common source, shared fresh control and unchanged
15-task diagnostic matrix: all 11 selected failures plus ring16 acquisition,
mode hold, stripes and Gaussian smoke. Gaussian stability restores each arm's
own passing smoke checkpoint. Keep original budgets, data sequences, seed 0,
public initialization, sampling and observation cadence, with one recipe
globally. Full scheduled gates determine PASS; an endpoint improvement does
not substitute for retention or target-shift recovery.

Prediction: wherever `0 < t < 1` and the base move is nonzero, actual G/prior
movement has L2 ratio `t` to the proposal within floating-point rounding; the
Gaussian/overlap terminal persistence improves without losing protected
control passes. Record uncooled gradient norm, thermal scale, base movement and
applied movement by role together in external traces. This is a measurement
of actual parameter motion; output movement and signed covariance/shape data
remain separate measurements in the common task artifacts.

Falsifiers: thermal scales stay one in the discriminating windows (the proposed
mechanism is inactive there); the movement ratio does not match its declared
law (implementation failure); smaller realized moves fail to improve Gaussian
or overlap retention; or any protected control PASS becomes FAIL. All original
scientific failures remain visible. Gaussian's all-stationary and post-shift
requirements cannot be relaxed because the model passes some observations.

Only the registered first round is authorized by this preregistration. A
substantive successor requires the saved first-round measurements and the
parent's explicit bounded declaration; there is no threshold sweep, seed
experiment, automatic continuation or adoption.

## Software checks and reproduction

The frozen `tests/test_bispf_settling_repair.py` suite had 21 passing checks: realized matrix,
bias and sampled-prior movement; untouched rows including signed-zero bits; unchanged gradients and
optimizer histories; exact uncooled execution; critic-hook and SBNP gradient
restoration; nested/exception/missing-cache behavior; no RNG consumption;
checkpoint continuation; and rejection before mutation for invalid scales or
critic optimizer groups. The checks also cover public `GANTrainer` equivalence,
cooled and zero motion, exact complete checkpoint/stream continuation, and
exception restoration. They are software
evidence, not toy-training quality results. The common measured readout must supply
public-API task receipts and actual-training GIFs before a research PR is
presented as a training solution.

The post-execution strict-progress composition guard adds one rejection check;
the current suite passes 22 checks. It does not change the executed
`direction_blend` recipe or its retained evidence.

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests/test_bispf_settling_repair.py
```

Bulk logs, per-update metrics and states belong under
`/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022`, outside Git. The
parent common-source runner will publish the exact `tail -F` path and execution
commands with the frozen study; no training command is fabricated here.

## Execution telemetry limit

The preregistration above requests per-update scale, base movement and applied
movement by role. The public helper returns those diagnostics when requested,
and the software controls verify its realized displacement law. However, these
Forge adapters do not retain the trainer's returned `thermal_displacement`
dictionary across updates. No complete trained movement-ratio trace is claimed
for this cohort. Final observer states and certified saved model tensors remain
available for the narrower comparisons in the measured readout.

The optional `OptimizerDiagnostics` observer attaches post-hooks to the
underlying `optimizer.step`, which runs before the helper interpolates the
parameters. Its G/prior norms would therefore describe **base proposals before
cooling**, despite its generic actual-step label. They must not be interpreted
as realized cooled movements. Persisting diagnostics after the helper would be
a separately versioned observer correction; it does not authorize rerunning
unchanged scientific experiments or changing the frozen cohort.
## Saved-state signal screen: reject an unchanged energy observer

The read-only analyses below add no inference draw or training update. They
use certified checkpoints and actual scored output tensors from the new campaign.

Baseline Gaussian checkpoints at steps 1000/4000/6000 have thermal scale 1 and
EMA nearest-neighbor residuals .15909/.10741/.17516, above unchanged tau .020.
Overlap's final scale is 1 with residual .23608. These sparse observations do
not establish the whole history. However, 160 disjoint 128-versus-128 panels
from Gaussian's passing saved outputs have same-law nearest-neighbor residual
minimum .02499 and median .05627; overlap's 160 panels have minimum .09373 and
median .12583. Every panel exceeds .020 even though both halves come from the
same generated law. This exposes a finite-sample floor in the existing signal.

One structural observer proposal was fixed before computation: use the signed,
normalized energy U-statistic
`[mean|R-F|-.5 mean_offdiag|R-R'|-.5 mean_offdiag|F-F'|]/RMS_offdiag(R-R')`,
with scale 1 for a constant real batch. Its positive part would replace the
raw nearest-neighbor residual. No signal threshold was picked or swept.

Actual real training batches were not saved. This **separate saved-data
diagnostic** therefore fixes references to the first receipt-PASS generated
state: Gaussian smoke 209, shifted Gaussian 4709 and overlap 100. These are
approximate empirical references, not historical real batches or an oracle
training mechanism. Thirty-two disjoint 128-row panels per state use a fixed
one-panel reference rotation, including at the reference state itself.

| Diagnostic | Smoke | Gaussian continuation | Overlap |
| --- | ---: | ---: | ---: |
| Receipt-PASS / FAIL states | 6 / 18 | 10 / 110 | 10 / 14 |
| State-mean signal AUROC, PASS versus FAIL | .99074 | .93818 | .85000 |
| AUROC versus states at least 2x an original gate bound | 1 | 1 | 1 |
| Median mean pointwise thermal scale on PASS states, tau .020 | .14384 | .12063 | .40410 |
| Median final scale after 32 fixed-state panels under original fast-attack/.95 EMA | .86307 | .76366 | .99934 |

The last row resets the EMA once per scored state and follows fixed saved-panel
order; it does not reconstruct historical training. Threshold-free separation
looks useful, but the unchanged peak detector reheats passing overlap states
almost completely. A less biased input alone does not solve that controller.

Explicit **analytical point-mass fixtures** additionally test premature cooling.
They are separate zero-width controls, never substitutes for the original
positive-width task laws. Exact population weighted distances remove sampled
estimation noise:

| Fixture | Mass TV | Normalized energy numerator | Unchanged tau .020 scale |
| --- | ---: | ---: | ---: |
| Four equal modes, one absent | .25 | .047420 | 1 |
| Rare 2% corner absent, mass moved to largest corner | .02 | .000681 | .003408 |
| Eight equal ring modes, one absent | .125 | .009069 | .430348 |
| Sixteen equal ring modes, one absent | .0625 | .001994 | .027855 |

Matched constant batches give energy 0 and scale .00001; distinct constants
separated by one unit give energy 1 and scale 1. Collapsing saved Gaussian or
overlap samples to their reference mean gives energy .16818/.19534 and scale 1.
Complete collapse remains detectable, while a rare missing population or one
missing ring component can be strongly cooled before acquisition is complete.

**Decision: do not implement or spend on energy-U as an unchanged drop-in
observer.** It would retain finite-batch peak reheating and introduce a clear
partial-acquisition risk, repeating the concern behind prior idle/capture
failures. Keep these negative controls. Finish round one, verify whether the
displacement candidate was numerically inert, and require a support-sensitive,
noise-aware observer proposal to pass saved-state controls before further
training. No successor, threshold tuning or extra round is declared here.

Reproduce the one fixed signal screen with raw per-observation scores outside
Git and an optional compact summary:

```sh
OMP_NUM_THREADS=1 python reports/forge/three-theme-repairs-20261022/probe_settling_signal.py \
  --output /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/settling-signal-probe.json \
  --summary-output /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/settling-signal-summary.json
```
