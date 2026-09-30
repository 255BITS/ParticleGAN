# A shared E22 bank at sequential token-routing sites

[`examples/e22_routed_sites.py`](../examples/e22_routed_sites.py) is a complete
caller-owned paired-error training loop with two sequential routing sites.
Both sites use one particle table, one mass vector and one E22 policy. The
second site's queries depend on the first site's output, so a structural
proposal must rerun the full model before measuring its final error.

The example supplies source, time and spatial position conditioning. It runs
two frozen BF16 host layers alongside FP32 encoder, query modules, adapters,
table and critic. It trains RpGAN with KA2 on paired residuals; it evaluates
clean output RMSE on a third disjoint source/time grid. Output MSE is not a
training loss or an acceptance condition.

## Complete model callback

Use the full-model alternative to the [single-site contract](e22_routed.md):

```python
rows = RoutedRows(model_forward=model_forward, features=paired_features,
                  sites=("first", "second"))
```

The callback receives an ephemeral `RoutedExecution` helper:

```python
import math

def model_forward(models, context, candidate, routing):
    G, E, R = models["generator"], models["encoder"], models["router"]
    encoded = E(context)
    first_logits = R.first_query(encoded) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    first_codes = routing.mix("first", first_logits)
    hidden = G.first(context, encoded, first_codes)
    second_logits = R.second_query(hidden) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    second_codes = routing.mix("second", second_logits)
    return G.second(hidden, second_codes)
```

The runnable callback obtains every query and host/adapter from `models`;
it does not capture live training modules. Each `mix` call centrally adds
the candidate's `log_mass`, applies softmax and mixes the shared table. Do
not add the mass again in the logits. Both keys and values differentiate
through the same table. The helper optionally applies DV12 independently to
the mixed codes at each site, preserving the native controller's limited
recent perturbation records.

Call each declared site exactly once in the declared order. Missing, repeated
or out-of-order sites raise an error. The helper expires when the full model
call ends; do not retain it or reuse cached training activations for a
counterfactual. Site two must recompute its queries from the candidate-dependent
hidden output. Independent DV12 draws at the two sites retain their sequential
ordering.

## Evidence and guards use the final conditioned model

Every deletion probe and split proposal reruns both sites with the proposed
bank and row state. Its learned feature error comes from the final model
output and the corresponding paired target. The example extracts the critic's
learned token residual features and flattens them into one feature vector per
conditioning context. It does not compare isolated site codes or hypothetical
independent row outputs.

Routing usage averages tokens within each context, then averages the declared
sites with equal weights. Extra spatial tokens or sites do not increase the
effective-context count. The conditional evidence remains a weighted empirical
diagnostic, without the original independent-particle BH or conformal claims.
The [routed law](e22_routed.md#conditional-evidence-and-coupled-moves) describes
its deletion effects, gradient persistence, mass split and per-context bounds.

Fitting contexts drive gradients and deletion diagnostics. Separate guard
contexts protect coupled fast and averaged proposals. A third context grid
measures final clean outputs and never enters the policy. The empirical guards
do not guarantee improvement on that final grid or on every possible
conditioning input.

## Running, restoring and serving

```bash
python -u examples/e22_routed_sites.py --steps 60 --output /tmp/e22-sites.pt
python -u examples/e22_routed_sites.py --steps 2 --resume /tmp/e22-sites.pt
```

The CLI uses the public API initializer by default. The former `initialize_`
entry point is now `init.deterministic_orthogonal_`, with the same values;
see the [migration table](api.md#migrating-from-initialize_-and-recipeinitialization).
Whole networks use the documented role keys G=0, D=1, E=2, with router=3.
The bank is initialized through `recipe.make_prior()` and the public R2
initializer before the frozen comparison arm disables its gradients. This
path has neutral row masses and no injected outlier.

Python `make_loop()` retains `initialization="conformance"` as its default
for the lifecycle tests. That deliberately constructed fixture has a
hand-built bank, constant query weights and a downweighted outlier. Select
it explicitly in the CLI with `--initialization conformance`; select
`initialization="api"` in Python for quality comparisons.

The default task uses 8 spatial tokens, a 16×2 bank and a batch of 8 contexts.
Inputs have shape `[batch, tokens, 4]` with columns source-x, source-y, time
and spatial position. Final outputs have shape `[batch, tokens, 2]`. The first
FP32 residual is cast at the next BF16 host layer; the final residual projection
and reported output remain FP32. Frozen parameters retain their BF16 dtype and
exact values throughout training, averaging, serving and recovery.

The lifecycle follows the [ordinary policy hooks](e22.md#caller-owned-updates).
The caller supplies `RoutedBatch` pairs and guards, generates with `sigma=0`
and adds shared noise in normalized paired-error coordinates. Real and fake
receive the same Gaussian base draw; the real reference is detached during
generator training. This caller-owned noise mapping differs from the generic
served helper's optional direct output noise.

The application checkpoint contains policy state, model shape/mode and two
application RNG states: batch selection and the shared paired-error base
stream. Policy state includes its own DV12 streams, row controls, full-model
site contract, controllers, noise, optimizer moments and coherent fast/averaged
weights. Restore at a completed update boundary. A resumed CLI invocation
rebuilds the stored shape and mode before loading.

```python
served = policy.served_model()
prediction = served.routed_forward(source_time_position)
```

This clean default uses frozen copies of both hosts, both query modules,
encoder, adapters and one consistent selected table/mass vector. All sites
use fast weights together or averaged weights together according to E22's
served-choice rule. Training modules remain fast and serving consumes no
training RNG. Optional `perturb=True` enables per-site DV12; `output_noise=True`
adds the stored sigma directly in prediction coordinates.

The fixed tiny CPU conformance fixture reduced clean final-grid RMSE from **.188005 to
.015629** after 60 actual adversarial updates. All 16 rows received nonzero
gradients each time. The controller accepted four splits (eight moved rows),
rejected 43 guarded proposals, and its active evidence held table descent for
one update. This is a synthetic integration result, rather than a diffusion
model benchmark.

Conformance tests cover downstream query recomputation, full token-output
features, context counts, active evidence and accepted moves, exact
CPU-deserialized resume, and coherent fast/averaged snapshots. The CUDA test
uses actual BF16 host operations with FP32 trainables and replays a naturally
accepted move after `torch.load(..., map_location="cpu", weights_only=True)`.
Exact replay is checked on the same device with serialized backward execution.

## Activation-checkpointed whole-model replay

[`examples/e22_routed_replay.py`](../examples/e22_routed_replay.py) wraps the
complete eager two-site generator forward, including the dependent second query:

```bash
python -u examples/e22_routed_replay.py --steps 8
python -u examples/e22_routed_replay.py --steps 8 --mode no_rows --device cuda
```

The example calls `update(loop, generator_forward=checkpointed_generate)`.
The ordinary critic update and paired-error noise stay in the outer loop.
Only the differentiable `policy.routed_generate(context, sigma=0)` is inside
the checkpoint. Each forward/recomputation creates a fresh `RoutedExecution`,
calls both sites in order and closes it; it never reuses the expired helper
from the first call.

Before the original forward, the wrapper captures
`policy.noise_generator.get_state()`. The original forward draws and records
DV12 normally. During recomputation, a context manager creates a separate
`torch.Generator` restored to that captured state and passes it as `stream`.
The same per-site draws reproduce the mixed-code perturbations. Because this
stream is not the policy's training stream, the existing policy disables DV12
diagnostic recording. Backward leaves the live training stream and controller
diagnostics at their post-forward values; it never rewinds the live stream.

The wrapper explicitly uses `use_reentrant=False`, `context_fn` and
`set_checkpoint_early_stop(False)` to replay the full model. PyTorch preserves
default CPU/device RNGs for operations such as dropout; the wrapper handles
the policy's explicit generator separately. See the
[PyTorch checkpoint contract](https://docs.pytorch.org/docs/2.6/checkpoint.html).

Keep inputs fixed and complete backward before changing module modes,
controller coefficients, weights or row state. Additional callback-owned RNG
streams need their own replay handling. The host layers in this example do
not update buffers.
Models that update buffers during forward must arrange for those updates to
happen once outside recomputation. Keep application logging outside the
checkpoint closure. Observations, policy hooks, optimizer steps,
evidence refresh, structural moves and serving averages all run once in the
outer lifecycle. Save recovery checkpoints after `finish_step()`; do not save
an in-flight activation graph or its replay context.

Tests compare exact outputs, gradients, RNG state, controller diagnostics and
actual paired-game updates with the ordinary two-site loop, including an
accepted row move, recovery and clean serving. Keep the `no_rows` movable-bank
baseline when validating this replay path and the Sliders integration.

## API-initialized spatial comparison

```bash
python -u examples/e22_routed_sites.py --compare --steps 40 --tokens 128 \
    --z-dim 4 --particles 128 --batch-size 8 --device cuda:1 --initialization api
```

This shape means a **128×4 bank and 128 spatial tokens**, with four-column
conditioning inputs and two-channel final outputs. It compares three modes:

| Mode | Bank | Row evidence and restructuring |
| --- | --- | --- |
| `frozen` | `requires_grad=False`; adapters and queries still train | Both off |
| `no_rows` | Movable bank | Both off; no row observations, probes or proposals |
| `full` | Movable bank | Conditional evidence and guarded birth/death active |

The script checks identical initial parameters, batch indices and paired
Gaussian base draws, plus the paired and DV12 RNG states after every update.
DV12 makes four ordered draws per update, each shaped `[batch * tokens, z_dim]`:
first/second sites during the critic half, then first/second sites during the
generator half. The paired base stream is separate from DV12 draws, so
controller activity cannot change which base noise the next batch receives.
Learned noise amplitudes and DV12 amplitudes can differ as the three games
evolve. This is one deterministic comparison, not a seed sweep or a search
for settings where `full` wins.

Each mode reports clean final-grid output RMSE, maximum token output error,
accepted moves and training wall-clock time. Timing includes backward,
optimizer work and every controller/probe/guard operation, and synchronizes
CUDA at each measured update. Each comparison arm first runs one untimed
warmup update (`--warmup 1`), then restores its complete initial checkpoint,
module modes and gradients. Model construction, warmup, final validation and
log printing are outside the measured interval. The report includes
`warmup_steps`; `--warmup 0` exposes first-use overhead. Compare the measured quality and cost directly:
passing a learned-feature guard alone establishes neither better output RMSE
nor a faster training loop.

Measured on one NVIDIA RTX A6000 with PyTorch 2.13.0+cu126, 40 updates per
mode, batch size 8 and one restored warmup update:

| Mode | Clean held-out RMSE | Maximum token L2 error | Training wall time | Accepted splits |
| --- | ---: | ---: | ---: | ---: |
| Frozen bank | **.167288** | **.615440** | .641 s | 0 |
| Movable bank, row controls off | .210221 | .756912 | .600 s | 0 |
| Full `e22_routed` | .210221 | .756912 | 2.044 s | 0 |

All modes started at clean RMSE .341401 and served fast weights. Initialization,
batches, paired-error base noise and DV12 draw schedules matched. Full controls
made 320 deletion probes, evaluated 312 proposals and rejected 32 at the guard;
none was accepted. Its output metrics exactly match the movable baseline.
The [API comparison receipt](e22_routed_api_results.json) records the source
revision, hashes and diagnostics. The measured full-loop cost is about 3.4
times the movable baseline here, without a quality improvement.

Forty updates are inside KA2's initial 799 pure-A calls. A fixed continuation
through 1,200 updates, retaining matched initialization, batches and noise,
gives this clean final-grid RMSE trajectory:

| Updates | Frozen bank | Movable bank, controls off | Full `e22_routed` |
| ---: | ---: | ---: | ---: |
| 40 | .167288 | .210221 | .210221 |
| 160 | .026650 | .017721 | .017721 |
| 400 | .012024 | .008815 | .008815 |
| 800 | .008263 | .006100 | .006100 |
| 1,200 | .014569 | .005540 | .005540 |

Full controls accepted no moves throughout this continuation. The movable bank
improves on the frozen arm at the later checkpoints; the row controls provide
no observed extra quality on this task. Final maximum token errors are .047025
for the frozen bank and .025185 for both movable modes. The third grid remains
outside training and proposal decisions. This is one fixed trajectory, with
no seed sweep, endpoint selection or convergence guarantee. The
[trajectory receipt](e22_routed_api_trajectory.json) includes every checkpoint;
its exploratory elapsed times include validation and shared-device effects,
so use the separate 40-update receipt for the cost comparison.

## Historical fixture and row-state diagnosis

The earlier comparison used the hand-built conformance initialization:

| Historical 40-update arm | Held-out RMSE | Maximum token error |
| --- | ---: | ---: |
| Frozen bank | .078747 | .179697 |
| Movable bank, controls off | .023803 | .068836 |
| Full controls, current row-state reset | .031008 | .075403 |
| Full controls, accepted commits suppressed (diagnostic) | .023803 | .068836 |
| Full controls, parent Adam moments inherited (diagnostic) | .023454 | .083046 |

The original [receipt](e22_routed_spatial_results.json) is retained with its
historical revision and hashes. It describes a deliberately constructed
lifecycle fixture, rather than the current API-initialized quality comparison.
The [diagnostic receipt](e22_routed_formulation_diagnostics.json) contains the
exact reproduction, accepted-move observations and two matched diagnostic
arms. The latter preserve batch and paired/DV12 noise traces; neither changes
the validation grid or uses it to select moves.

Suppressing actual commits while keeping probes and proposal evaluation active
reproduces the movable baseline exactly. Inheriting the parent's Adam moments
for both moved rows removes the mean-RMSE disadvantage, with the other
controller/history resets retained. Its maximum error worsens, so this is a
causal diagnostic, not a qualified replacement law.

The current routed commit clears moved-row first/second moments and AMSGrad
maxima while keeping the optimizer's shared step counter. With beta1=0 and
beta2=.999, the next nonzero row update after a late reset can be roughly
5.7–6.3 times a fresh Adam update at the observed steps 32–39, before other
controls. Tiny immediate gains can therefore precede substantially different
subsequent learning. A coherent split-state transport or row-local optimizer
age needs a late-step regression that checks the next update's size. For exact
half-mass duplication, halving first moments and quartering second moments is
a mathematical starting point to validate, rather than a proven rule for all
coupled splits.

There is also a metric mismatch. At step 32, an accepted move lowered guard
feature error from .0618101 to .0617447 while increasing guard RMSE from
.0688192 to .0688366 and held-out RMSE from .0666459 to .0666674. Step 35
showed the same disagreement. A learned-feature guard protects its own metric;
it does not guarantee lower raw output error. Applications that require paired
output accuracy should consider an additional output-error guard on protected
contexts, leaving the final validation grid untouched.

Exact duplicate proposals preserve parent mass after retiring the child and
add no distinct routing code; identical cloned rows remain symmetric under
gradient updates between structural moves. Counts labeled `splits` include those replacements and
should not be read as an equal number of new independent routing components.

The RpGAN signs match the relativistic paired objective. The
[R3GAN analysis](https://arxiv.org/html/2501.05441v1) establishes local
convergence under its assumptions, without qualifying this extra row-state
transport or guaranteeing monotone output RMSE. Keep the movable baseline,
both mean and maximum output error, and complete trajectories in Sliders
validation. The initialization correction fixes the comparison setup; the
routed optimizer-state and guard questions remain ParticleGAN work.
