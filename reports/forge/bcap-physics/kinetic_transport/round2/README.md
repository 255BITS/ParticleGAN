# Round 2: relative local density moments

**The local density residual converts unequal mass to a complete sustained PASS.**
The successor passes **3/5 mutually runnable tasks**, versus **2/5** for its
matched transport-v1 primary control. Both arms additionally retain one explicit
two-pole BLOCKED cell. All ten runnable jobs finish once for **326.971374 paid
worker seconds**, with zero scientific retries and no remaining reservations.

## Completed matched results

| Unchanged task | Transport-v1 primary control | Local-density successor | Exact retained outcome |
| --- | --- | --- | --- |
| Two-pole | BLOCKED | BLOCKED | Both frozen component hosts lack the sample-space signal consumer |
| Gaussian smoke | PASS, 13/24 confirmed states | PASS, 13/24 confirmed states | First confirmation 84→167; endpoint KS .026218→.020606 |
| Gaussian stability | FAIL, 33/72 stationary and 12/24 shifted hold | FAIL, 28/72 stationary and 11/24 shifted hold | Both miss deadline reacquisition; final KS .090741→.060653 still exceeds .05 |
| Unequal mass | FAIL, 11/24 checks, terminal suffix 2 | **PASS, 13/24 checks, terminal suffix 5** | Full covariance .773642→.522577; minimum mass ratio .961538→.899564; all five terminal gates now pass |
| Unequal width | FAIL, 0/24 checks | FAIL, 0/24 checks | Full covariance 7.038571→2.480438 meets the forecast≤3.5, but still fails the original≤.85 gate |
| Two broad | PASS, 24/24 checks, suffix 24 | PASS, 24/24 checks, suffix 24 | Full covariance .244338→.226564; mass TV .014160→.000732; passing guardrail retained |

[Exact metrics and temporal failures](results.json),
[trained-source and matched-condition receipts](provenance.json),
[deterministic center/data diagnostics](saved-state-diagnostics.json), and
[unchanged predecessor parity](predecessor-parity.json) support this readout.
The five primary-control trajectories, endpoint metrics, graders and named
final RNG states exactly reproduce the archived transport-v1 cohort. They are
new matched receipts under the current source, with no archived qualification
credit. The original BCAP winner is contextual rather than a third causal arm.

The unequal-mass last-five minimum eigen ratios are
`.478403, .159678, .442112, .486961, .366567`; the retained floor is `.15`.
Control fails at 1050/1100 with `.084535/.096804`. The update 1050 candidate margin
is small, so this measured terminal suffix does not establish wider robustness.
Saved centers change `144/73/34/5`→`141/77/32/6`. Rare served mass is
114/4096 (`.027832`) versus 96/4096 (`.023438`) for target `.02`; allocation
remains sufficient but slightly overshoots. Overall mass TV worsens
`.013496`→`.016426`, despite the new full quality pass. The gain concerns
sustained local variance and fidelity, rather than every endpoint mass metric.

Unequal-width centers remain almost unchanged: `62/63/65/66`→`61/64/65/66`.
The first two full component covariance errors fall from
`18.292221/9.450035` to `4.787480/4.601254`, but remain far above the original
bound. Their reported spill fractions increase from `.035491/.060181` to
`.042753/.062749`. Thus covariance reduction cannot be described as removing
stray events. Four-sigma core-average covariance worsens `.248193`→`.289171`,
and HQ falls `.974609`→`.955078`; neither endpoint improvement is universal.
These certified four-sigma core/spill summaries differ from the explicitly
three-sigma diagnostic cohorts used in the preflight above.

The candidate's Gaussian endpoint mean error `.071086` and width ratio
`1.039455` pass their individual bounds, while target KS `.060653` fails.
Stationary/shift pass counts regress, and the additional frozen expectations
≥33/72 and ≥12/24 are falsified. The unequal-width half-spill forecast and
rare minimum mass ratio≥.5 are observed; the broad guardrail passes. Automatic
study decisions remain **incomplete** because two-pole is unsupported, despite
all runnable training being complete. Numerical task gates are unchanged.

**Retain the opt-in local-density mechanism as a scoped sustained rare-variance
repair; stop this exact revision as a global repair.** Inspect remaining narrow
spill distances and Gaussian empirical-field fluctuations before any separately
preregistered successor or broader matched comparison. This completed round
launches no further tuning, candidate, continuation, seed run or promotion.
Full unequal-width and continuous-learning failures still block a global claim;
native, image, conditional and arbitrary high-dimensional transfer remain
unmeasured. A finite real-anchor witness can miss remote outliers and can trade
local shape against shared-map motion.

## Source, cost and validation

Executed scientific commit: `5c2a64682b8900c5b42de94a6c27502e41d500f2`.
Both arms execute source digest
`a22f9f4fe82b9d932f1dad635793d993708e155905b50fb090f758f844b65b09`.
Candidate revision: `4e91e9f5ad0cf89e14b2f4e73538693341bd9df1a591221cf5af15f8a6aff188`;
primary-control revision: `10116a6345f68563f4193750613049e9cdd28965409b5be2fdba554946afaaae`.
Publication later changes only reports and reproduction sources; no later
unmeasured trainer correction is presented as trained evidence.

Python 3.12.13, Torch 2.14.0, NumPy 2.5.2, RTX A6000, deterministic execution,
one Torch thread, TF32 off. Initial model/prior tensors, same target batches,
complete consumed named RNG states, prior/sampling laws and budgets match
between arms. Only effective local weight 0→1 differs. Both Gaussian continuations
restore their own passing smoke exactly, including all streams and prefix steps.
All 1200 vector batches replay to the checkpointed data stream exactly.

Each arm adds 9600 outer updates, 19200 total. Paid workers:
candidate 168.815268 + control 158.156105 = 326.971374 seconds.
Declared ceiling 12840, executed full reservations 12240, remaining reservation 0,
scientific retries 0, track allowance 14400. The bounded local drain finishes and
leaves no watcher. Shared-device contention is included; no speed claim follows.
Read-only field probes add no training/sampling and take a separately recorded
CPU budget below one second. Software/media work is outside the paid worker
ledger and supplies no scientific qualification.

[109 distinct meaningful software checks](software-verification.json) pass,
including exact empirical null/gradient, units/permutation/RNG invariance,
contraction direction, duplicate-neighbor floor, public-trainer consumption,
unchanged critic/streams, default checkpoints and unsupported-control admission.
The final transport/study check set passes 52 checks, overlapping the initial 108.
Forge declarations validate. [Scorer controls](scorer-controls-reuse.json) reuse
four oracle PASS/four collapse FAIL after verifying both unchanged scorer source
hashes, task laws and gates; no unchanged scientific training is rerun for them.
Publication reproduces **432 saved primary metric sets exactly** and renders
**10 actual-training GIFs**, each with nine fixed-index frames. Complete logs,
per-update events, saved arrays and checkpoints stay in this track's local queue.

| Task | Primary control actual-training GIF | Candidate actual-training GIF |
| --- | --- | --- |
| Gaussian smoke | [Target and draws](control-gaussian1d_smoke.gif) | [Target and draws](candidate-gaussian1d_smoke.gif) |
| Gaussian stability | [Hold and shift](control-gaussian1d_stability.gif) | [Hold and shift](candidate-gaussian1d_stability.gif) |
| Unequal mass | [Density and gates](control-vector_unequal_mass.gif) | [Density and gates](candidate-vector_unequal_mass.gif) |
| Unequal width | [Density and gates](control-vector_unequal_width.gif) | [Density and gates](candidate-vector_unequal_width.gif) |
| Two broad | [Density and gates](control-vector_two_broad.gif) | [Density and gates](candidate-vector_two_broad.gif) |

[Media receipts](media.json) bind each GIF to saved observation bytes. Reproduce
this readout without training or model sampling:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python reports/forge/bcap-physics/kinetic_transport/round2/publish.py
```

The declaration, mechanism, forecast and stopping contract below were pushed
before these runs and retain their original values.

This authorized successor retains transport-v1 as its **one primary matched
control**. It adds local real-anchor density moments to the existing allocation
signal. The [round-one report](../README.md), forecasts, receipts and failed
revision remain unchanged evidence. The original BCAP winner supplies historical
context, not a third matched arm. This is a mechanism diagnostic; no ordinary
qualification, calibration or public-default credit follows.

## Saved-state motivation

[Read-only field diagnostics](prior-field-diagnostics.json) consume certified
round-one output arrays and the final saved critic. Each consecutive 128-output
chunk is paired with the exactly replayed last target training batch. These are
posthoc **output-space probes**, not reconstruction of the original G-phase
latent rows, pullbacks or optimizer proposals. They add no optimizer updates or
generated draws. Target labels and Mahalanobis radius≤3 define core/spill only
for this diagnostic; none enters the training signal.

Unequal-width projected pairings cross nearest-component labels in **23.9075%**
of pairs. Narrow spill components 0/1 have transport/critic force cosine means
**−.861/−.798**. Their transport mean outward radial forces are
**−.003302/−.003474**, while adversarial forces are **+.002617/+.002042**.
Thus the measured transport already pulls many stray points inward; projection
mismatch alone cannot explain spill. Shared-map response and opposing critic
motion remain plausible. An added local feature force contributes inward
**−.000109/−.001036** on these narrow spill cohorts, but contributes outward
**+.008048** in the fourth component's core. This mixed preflight is a
falsifiable lead, not evidence that all covariance will improve.

Round one's rare census improves 1→5 and minimum mass ratio reaches .961538,
but variance misses at 1050/1100 break the terminal suffix. Unequal-width
occupancy becomes near-equal while full covariance error is 7.038571 versus the
.85 gate. Local shape and spill therefore require a signal beyond occupancy.

The archived [MMD witness preflight](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/mmd-witness-preflight/README.md)
did not justify its standalone local-width online candidate: decreases mostly
came from self-repulsion; matched advantage remained unresolved on failing
native tasks. This successor retains a separately measured allocation signal,
uses real-anchor relative residuals rather than unnormalized Gaussian U-MMD,
and explicitly risks the same excessive-spreading failure. It does not
reinterpret that preflight as a pass.

## Mathematical mechanism and trainer delta

For the same consumed real batch `y_1,...,y_n`, let `h_j²` be the squared
distance from anchor `y_j` to its fourth other neighbor (`min(4,n−1)` for small
batches). Duplicate neighborhoods use the explicit numerical floor
`epsilon * max(mean centered real coordinate variance, epsilon)`.
All anchors, widths and normalizers are detached. For fixed scales
`a∈{1,2,4}` define

`phi_(j,a)(x) = exp(−||x−y_j||² / (2 a² h_j²))`,

`p_(j,a) = (1/n) sum_l phi_(j,a)(y_l)`,
`q_(j,a) = (1/n) sum_i phi_(j,a)(x_i)`.

The global candidate is

`L_G = L_adversarial + L_SW2_normalized + L_local`,

`L_local = (1/(3n)) sum_(j,a) [(q_(j,a)−p_(j,a))/p_(j,a)]²`.

Every `p>=1/n` because its anchor contributes a unit value. For fixed real
anchors this is a positive semidefinite finite-feature MMD with kernel
`k(x,x')=(1/(3n)) sum phi_(j,a)(x) phi_(j,a)(x') / p_(j,a)²`.
It has zero value and zero fake gradient at equal empirical batches. It is not
a characteristic population kernel, unbiased estimator, or fitted density-ratio
method: the features depend on the same finite real batch. Self terms remain
explicit, and no held-out/calibration data enter training.

Writing `r=q−p`, the fake gradient is

`grad_x_i L_local = −2/(3n²) sum_(j,a)
  [r_(j,a) phi_(j,a)(x_i) / (p_(j,a)² a² h_j²)] (x_i−y_j)`.

Negative residuals attract toward underfilled real neighborhoods; positive
residuals repel from overfilled neighborhoods. Relative normalization reduces
squared-mass attenuation: an anchor in a sparse region is measured relative to
its own empirical density. The same generator/prior Jacobians pull back this
field; their normalized optimizer response is unchanged. The physical analogy
is density redistribution with a local compressibility residual, not a
continuous-flow or convergence guarantee. Far from all anchors Gaussian
features can vanish, so the retained global SW2 field still supplies transport.

[MMD, Gretton et al. 2012](https://www.jmlr.org/papers/v13/gretton12a.html)
provides the kernel mean discrepancy interpretation.
[Generative moment matching, Li et al. 2015](https://proceedings.mlr.press/v37/li15.html)
establishes differentiable kernel distribution matching in generator training.
Our adaptive relative real-anchor features are a particular finite witness;
these papers do not establish its GAN performance or guarantee density gates.

The only incremental field is `kinetic_transport_local_weight: 0→1`.
Transport weight 1, 32 directions and the complete winning base recipe remain
fixed: non-saturating, full DualNorm momentum 0, smoothing .001, per_offset,
G/E .012, D .018, prior .030, fixed BCAP coefficient/cap 1/every update,
floors 1, no input/output noise, EMA or prior regularization.
The public `Recipe.kinetic_transport_local_loss` and `GANTrainer` consume the
same already drawn G fake and real tensor. Frozen Forge adapters reuse that
per-update real tensor for D/G. No sampling, labels, target centers, prior
widths/weights, additional RNG draws or persistent state are introduced. Zero
local weight retains the exact predecessor trainer expression and recipe/
checkpoint default projection. Unsupported component hosts explicitly refuse it.

## Frozen comparison, forecast and stopping rule

Two ready studies share campaign `kinetic_transport_round2`, one candidate
`kinetic_transport_local_v2` and one matched primary control
`kinetic_transport_sliced_v1`. Both run on identical current source/runtime,
protocol seed 0, public deterministic initialization and unchanged full tasks:
Gaussian smoke and its own dependent stability; unequal mass; unequal width;
and the passing two-broad vector guardrail. Both two-pole cells remain explicitly
BLOCKED because the frozen fixture lacks the sample-space signal consumer.

All vectors retain 1200 updates, 24 checks, 4096 clean/live draws, five passing
terminal checks, full covariance≤.85, minimum eigen ratio≥.15 and all other
original bounds. Gaussian retains its full acquisition, stationary, deadline
reacquisition and postdeadline hold gates. A failed smoke blocks only that arm's
own continuation. No archived control fills these new-source cells.

Preregistered Forge signature uses the actual receipt key:
**unequal-width `component_covariance_error <=3.5`**; `>3.5` falsifies the
half-spill forecast. This diagnostic signature is weaker than the original
.85 numerical gate; it cannot create a task pass. Additional frozen expectations:
rare `min_mass_ratio>=.5`, broad-vector full PASS, Gaussian stationary passing
checks ≥33/72 and shifted hold ≥12/24, with full sustained grades authoritative.
There is no forecast revision, lambda/direction sweep, extra seed, additional
candidate or automatic third round.

The competing explanation is finite-anchor variance, rare-mode absence, amplified
relative residuals and shared normalized pullbacks. Local features can miss
remote spill or widen already sufficient clouds. Empirical training loss reduction
cannot establish held-out fidelity. Stop this exact candidate after this one
full comparison regardless of the outcome, publish the remaining failures,
and recommend only work justified by the measured evidence.

Declared full ceiling: 6420 seconds per arm, 12840 total, below the fresh 14400
track allowance. Both blocked 300-second cells consume no reservation, so full
runnable reservations total 12240 if both smoke prerequisites pass. Saved CPU
probes and meaningful software checks are separately costed within the ceiling.
No capacity probe or training diagnostic is needed. Worker contention is cost
accounting, not optimizer-speed evidence.

## Execution and reproduction

Revised mechanism and ready declarations are pushed before enqueue. Both arms
use this track's own queue and one GPU worker. The local public Queue/drain
runner disables the full-compile completion callback. Raw logs/checkpoints stay
outside Git; actual-training GIFs and compact receipts will be published here.

```sh
QUEUE=/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/kinetic_transport/queue
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root "$QUEUE" plan kinetic_transport_local_v2 --study kinetic_transport_candidate_round2 --show-boundaries
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root "$QUEUE" enqueue kinetic_transport_local_v2 --study kinetic_transport_candidate_round2
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root "$QUEUE" enqueue kinetic_transport_sliced_v1 --study kinetic_transport_control_round2
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/kinetic_transport/queue/kinetic_transport_round2/progress.jsonl
```

Run the read-only field probe with project Python 3.12:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python reports/forge/bcap-physics/kinetic_transport/round2/diagnose.py
```

The primary control's new-source results isolate this incremental mechanism.
Winner/round-one results remain contextual evidence under their original source.
Native, image, conditional and arbitrary high-dimensional transfer are unmeasured.

Study admission now records a known public-API refusal on either arm, including
the primary control, while retaining unsupported cells and their frozen task
identities. This repairs the previous whole-study refusal for transport-v1 as
control. It does not authorize an unsupported host or change a worker/gate.
The regression checks both BLOCKED bindings and the supported local-weight delta.

The runner uses physical GPU index `"1"`, one worker, sharing enabled:

```python
from pathlib import Path
from experiments.forge.queue import Queue, drain
queue = Queue(Path("/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/kinetic_transport/queue"),
              report_root=Path.cwd() / "reports/forge", on_completion=None)
drain(queue, ["1"], workers_per_gpu=1, allow_sharing=True, watch=False,
      campaign="kinetic_transport_round2")
```

An initial coordinator invocation supplied `cuda:1` where the public queue
requires a physical numeric index. It failed before any claim or worker;
correcting only the local runner launches the original two frozen requests.
This is not a scientific retry or source revision.
