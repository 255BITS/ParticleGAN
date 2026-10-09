# Round 2: relative local density moments

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
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/kinetic_transport/logs/coordinator.log
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
