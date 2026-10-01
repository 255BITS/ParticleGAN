# Conditional dense-bank E22: `routed_paired`

`get_recipe("e22_routed", ...)` selects a named adaptation for conditional
dense banks. It keeps E22's stationarity learning rates, anchored release,
DV12, learned noise, row controls, checkpointing and served averages. Its row
evidence and structural decisions use conditional counterfactuals rather than
the independent-particle statistics in [`e22`](e22.md). It makes no claim that
the original equal-mass density, split-conformal or Benjamini–Hochberg laws
apply to a dense conditional blend.

The [penalty, represented-mass and noisy-game guide](e22_routed_game.md)
documents the current DV12 law, token penalty convention and checkpoint
compatibility. Structural evidence remains a clean paired-feature diagnostic;
its instantaneous gains do not establish adversarial game repair.

The runnable [`e22_routed_paired.py`](../examples/e22_routed_paired.py) trains a
paired-edit task using source and time inputs. A frozen BF16 host runs beside
an FP32 encoder, router, adapter, table and critic. The source/time context
chooses a dense mixture; the same table supplies both routing keys and values:

```python
q = encoder_and_router(context)
weights = softmax(q @ table.T / sqrt(z_dim) + log_mass)
codes = weights @ table
prediction = frozen_host(context) + adapter(context, codes)
```

This resembles the integration's down/routed/up projection: the trainable
projection consumes FP32 features, and casts at the frozen host boundary are
the application's responsibility. A policy accepts mixed frozen precision;
it requires trainable parameters and the table to share dtype and device.
Frozen host tensors keep their original dtype and exact values in the fast,
averaged, checkpointed and served models.

## Binding the conditional forward

```python
from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe

recipe = get_recipe("e22_routed", num_particles=16, z_dim=2,
                    batch_size=32, output_noise_std=.125)
rows = RoutedRows(route=route, generate=generate, features=paired_features)
policy = E22Policy(
    recipe, G, D, table=table, encoder=E, router=R,
    generator_optimizer=opt_g, critic_optimizer=opt_d,
    roles=[["generator", "encoder", "router", "table"], ["critic"]],
    routed_rows=rows,
)
```

Supply three stateless callbacks. Each receives an explicit model mapping;
callbacks must use those supplied modules so the same code evaluates frozen
serving copies and counterfactual proposals.

| Callback | Contract |
| --- | --- |
| `route(models, context, candidate)` | Return the full dense routing weights for every context and row, using `candidate.table`, `candidate.log_mass` and any declared `candidate.row_state`. |
| `generate(models, context, candidate, weights)` | Return the full conditioned output, decoding `candidate.codes`. The spec computes `weights @ candidate.table` and optionally applies DV12 to these mixed codes before calling the decoder. |
| `features(models, context, samples, targets)` | Return deterministic learned critic features of the paired residual. Targets map to the real, zero-error reference. Preserve context order and use no RNG. |

The example's features are `D.features((samples - targets) / D.scale)`, where
the scale comes exclusively from fitting pairs. These features match the
critic's normalized error coordinates. A general integration can implement
its own paired residual transform and learned critic feature extraction.
The native fields `birth_death_feature_scale`, `birth_death_isolation` and
`row_evidence_null` do not select their independent-row statistical laws in
this adaptation. `RoutedRows.features` defines the conditional metric and
its explicit effect, persistence and guard bounds define the row decisions.

Register `router.log_mass` as a row-indexed buffer. Additional row-local router
parameters or buffers must be declared using `row_parameters=` and
`row_buffers=` on `RoutedRows`. They are part of each candidate and move with
the corresponding table row. Shared router/encoder parameters are not
row-indexed state. Undeclared mutable callback resources belong to the caller
and require separate checkpointing.

Tied key/value routing differentiates through both `q @ table.T` and
`weights @ table`. DV12 acts on the mixed code after routing, leaving the keys
unchanged; it is a conditional adaptation of the perturbation placement.
Decode `candidate.codes` directly. Recomputing `weights @ table` inside the
decoder would lose that perturbation.

Routed DV12 bandwidth uses mass-weighted atom spread and effective atom count;
support clipping excludes zero-mass rows. Exact duplicate positions sum their
mass, so replacing an inactive row with a half-mass duplicate preserves clean
and noisy forwards under identical draws. Independent-row DV12 is unchanged.

## Paired-error lifecycle

Use the same [critic/backward/generator/finish order](e22.md#caller-owned-updates)
as the independent policy. Supply fitting pairs and separate proposal guards
at the start of each update:

```python
batch = RoutedBatch(fit_context, fit_targets, guard_context, guard_targets)
noise = policy.begin_step(fit_targets, routed=batch)
prediction = policy.routed_generate(fit_context, sigma=0, perturb=True)
epsilon = noise.output_sigma * randn_like(prediction)
real_error = epsilon
fake_error = epsilon + paired_residual(prediction, fit_targets)
```

The example uses RpGAN with KA2 on these paired errors. Both halves receive
the same noise vector; `sigma=0` avoids an additional fake-only noise draw.
During the generator half, the real path is detached and the fake path
retains the learned noise scale's gradient. The caller invokes
`observe_critic_pair(real_error, fake_error)` and the ordinary lifecycle hooks.
No output-space MSE is used as a generator loss. A caller can compose
additional application losses, but should report that change in its task.
Here the .125 scale is measured in normalized paired-error coordinates;
it is the caller's shared error noise, rather than an extra perturbation of
the reported host prediction.
The noise scale ordinarily uses the physical `torch.maximum` and its existing
gradient, including the full above-floor gradient in FP64. The repair applies
only when rounding gives `exp(log_sigma) < floor` while
`log_sigma >= log_floor`, as can happen at CUDA initialization with .125.
That branch uses the log-space derivative, including the half subgradient at
a tie, with a detached correction preserving the exact physical floor.
As in [native E22](e22.md), the noise group's own stationarity scale does not
gate floor release. The generator, encoder, router and table scales must all
reach `1/64`; the floor then follows controller mobility, letting previously
clamped noise learn again. The noise group's learning-rate control still runs.

Three context sets have different jobs:

1. Fitting pairs train the generator/encoder/router/table and critic, and feed
   the conditional diagnostic reservoir.
2. Separately supplied guard pairs receive no training gradients. They protect
   actual coupled proposals using both fast and averaged models.
3. A final disjoint context grid is used only to report clean served RMSE.
   It is never supplied to the policy, feature guards or proposal selection.

The guard set is a controller input, so its performance is not an untouched
validation result. The final test grid is the held-out validation in this
example.

## Conditional evidence and coupled moves

Deleting a dense row changes the normalization of every route. The routed
controller therefore evaluates full conditional deletion counterfactuals on
the fitting reservoir. It combines their learned-feature residual effects
with route-weighted observations and gradient persistence. Its diagnostics
describe this named law; they are not independent-row p-values.
The weighted effective-context count describes the current fitting reservoir
at a row's latest probe. Replaying a reservoir does not increase the count;
repeated contexts within it still are not independent observations. The
count carries no false-discovery or split-conformal validity claim.

A proposal retires a child site and splits a selected parent's mass. The
parent and child receive cloned row state and their `log_mass` values are
reduced by `log(2)`. A bounded table split can separate their tied key/value
rows. The proposal's full conditioned forward is evaluated before mutation,
including routing, decoding and the learned critic residual. Separate guard
contexts must accept the change for both the fast and averaged banks.
These are empirical checks on the supplied contexts and current learned
features. They do not prove safety over every possible conditioning input or
guarantee improvement on the untouched final grid.

Purely duplicating a parent and halving its mass preserves that parent's
represented component. Retiring an existing child can still affect all
routes, and a perturbed split can change both keys and values. Mass accounting
alone cannot establish safety; the coupled feature guards make that decision.

On acceptance, table and declared row-local router state change together in
the fast and averaged banks. Both split rows inherit half the parent's Adam
first moment and a quarter of its second moment and AMSGrad maximum; the
shared optimizer age remains unchanged. Other rows retain their history.
Structural row owners must use Adam, AdamW or native `K3PGeneratorAdam`;
other optimizers remain usable when birth/death is disabled. This transport
avoids clearing moments while retaining an old bias-correction age. Its
half-mass gradient interpretation is exact for an unperturbed duplicate
without coupled regularization. Retiring a nonzero child, perturbing the
split or coupling regularization makes it a history prior, not a guarantee
of future output quality.
The [late-step regression](../tests/test_e22_routed_optimizer_transport.py)
reproduces a 29.517× next-update amplification at Adam age 2,048 under the
former reset, and an exact next-update match (maximum error 0) to the correctly
aged half-gradient reference under the revised E22 Adam transport.

Optional latent directional history for the changed parent/child resets
because their geometry can change. Conditional evidence resets globally
because softmax normalization couples all rows. The policy rebases affected
stationarity histories.

The example sets the diagnostic budget and guards explicitly:
`probe_budget=8`, `reservoir_size=64`, `min_observations=8`,
`min_effect=1e-8`, `improvement_margin=1e-10`,
`max_context_harm=1e-4`, `persistence_threshold=.75`, `split_scale=.1`.
These values belong to its small, normalized paired task. The positive mean
improvement margin and the per-context harm bound are separate conditions.
In particular, `1e-4` is the example's explicit benchmark override, not the
zero `RoutedRows.max_context_harm` default. The token-site API-initialized
example now defaults to zero and records any allowance in its checkpoint.

`RoutedRows(probe_interval=K)` spaces expensive deletion, proposal and guard
passes by at least K observed training updates; its default is 1. It counts
successful `begin()` observations, independently of batch size and
`min_observations`. A pass needs both the update interval and enough paired
contexts. Skipped intervals retain the context observations. Gradient
persistence and cheap evidence updates still run each training update.
Evidence-only controls follow the same interval without proposing moves or
reading protected errors. Disabled row controls collect no row observations
and do not advance this clock.

Checkpoint `probe_clock` stores observed updates and the last eligible probe
update. Interval 1 is omitted from the configuration for compatibility; old
default-interval checkpoints without a clock restore with a canonical zero
clock. A nondefault interval is part of the configuration and requires its
stored clock on resume. Interval selection changes controller timing, so
measure its quality and cost rather than assuming less frequent probes are
equivalent to the default trajectory.

An optional paired-output guard supplements these learned-feature checks:

```python
rows = RoutedRows(route=route, generate=generate, features=paired_features,
                  output_error_guard=True,
                  max_output_error_increase=0.0,
                  max_output_context_harm=0.0)
```

On the separate protected contexts, it computes raw final-output paired MSE
in prediction coordinates, without the critic's residual normalization,
averaging all nonbatch dimensions within each context. It requires both the
mean increase and the largest context increase to stay within the specified
bounds, separately for fast and averaged proposals. The learned-feature
guard must also pass. Output MSE neither trains the model nor selects
proposals or supplies row evidence; fitting proposals still use learned
features alone. A rejection leaves live weights, serving averages and
optimizer histories unchanged, while recording the failed guard.
Enabled diagnostics report `feature_guard_accepted` and
`output_guard_accepted`; `accepted` is their conjunction.

The output guard defaults off to retain the existing row law and checkpoint
configuration. Enabled settings are serialized and must match on restore.
With evidence-only or disabled row controls it performs no protected output
measurements. Zero bounds prevent worsening on those supplied contexts at
the current weights; they do not promise future optimizer behavior or better
error on the untouched final grid. The [multi-site CLI](e22_routed_sites.md)
exposes this option as `--output-error-guard`.

## Recovery, clean serving and validation

```bash
python -u examples/e22_routed_paired.py --steps 160 --output /tmp/routed-e22.pt
python -u examples/e22_routed_paired.py --steps 20 --resume /tmp/routed-e22.pt
```

Each JSON log row includes losses, dense table gradient coverage, learned
noise and structural diagnostics. The final row reports initial and final
held-out RMSE from clean frozen serving. This is one fixed deterministic task,
with no seed sweep.

The historical record at source revision `8618b19c` used the former
zero-moment split transport. On its fixed CPU task with 16 particles, batch
size 32 and 160 updates:

| Served model | Held-out RMSE | Held-out maximum vector error |
| --- | ---: | ---: |
| Initial | .178877 | — |
| Trained, clean fast weights | .005134 | .011593 |

All 16 table rows received nonzero gradients at every update. The controller
accepted 12 splits (24 moved rows), rejected 87 guarded proposals and made
1,280 deletion probes. Row evidence flagged persistent forces and held table
descent for four updates. The learned log-noise parameter received gradients;
the served scale remained at the task's .125 floor. These numbers describe
this one synthetic paired task, rather than a result on a diffusion or slider
model. Use this example as an integration check and validate a real task on
its own untouched contexts.

At completed update boundaries, save `policy.state_dict()` together with the
application's sampling RNG/cursor. The state includes the conditional
reservoir, routed evidence, guards, private structural RNG, declared router
state, table moments, controllers, noise and fast/averaged weights. Rebuild
the same callback contract and model/optimizer layout before loading.

```python
served = policy.served_model()
prediction = served.routed_forward(source_and_time)
```

`routed_forward` uses independent frozen copies of G/E/router/table and the
current fast-or-average served choice. Its default is a clean conditional
forward with no DV12 or output noise and no training RNG consumption. An
explicit `perturb=True` requests mixed-code DV12, and `output_noise=True`
requests direct prediction noise. Independent-row `sample()` is invalid for this conditional
bank.
The generic `output_noise=True` option adds the stored sigma directly in
prediction coordinates. A task using normalized error-coordinate noise,
as this example does during its paired game, must apply its own inverse
residual transform when it needs a matching stochastic prediction. The
reported validation and the example's serving path use the clean default.

Those measurements are retained as a historical integration record, rather
than a benchmark of the revised transport or optional output guard.

The conformance tests train this caller-owned paired game, require dense table
gradients and active evidence/restructuring, and restore immediately before a
naturally accepted move. They compare resumed losses, all policy state,
structural diagnostics, held-out metrics and clean served outputs exactly.
The independent native E22 conformance tests remain separate.

For multiple sequential token sites sharing one bank, see the
[complete-model example and matched comparison](e22_routed_sites.md). Its
second-site queries depend on the first-site output, and every conditional
probe or guard reruns the full conditioned model.
