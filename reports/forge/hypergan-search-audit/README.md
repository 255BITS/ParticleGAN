# HyperGAN / Hyperchamber audit of Forge configuration search

The recommendations are now implemented in this PR. See the
[implementation readout](IMPLEMENTATION.md) and
[implementation guide](../../../docs/forge-search-spaces.md) for public role
settings, historical loss/Adam variants and bounded categorical search
compilation. The findings and receipts below retain the original audit scope
at `0115a92f`; reproduce its software script from audit commit `f3100080`.
They do not describe the newly added interfaces or confer trained qualification.

Audited 2026-10-05 against `develop` commit
`0115a92f68dbf9bdcdf9e4f7bfea0fff8606e752`, in a separate feature worktree.

**Take Hyperchamber's compact configuration-selection idea into Forge as a
small declaration compiler. Keep Forge's evidence, ownership, budgets and
qualification machinery.** Forge already represents finite sets of numerical
Recipe choices, including coupled choices. It lacks random subset selection
and a common front end for categorical formulation choices. The optimizer and
loss in `halloween.json` are useful hypotheses, but **cannot be faithfully
represented by one current Forge Recipe**.

This is a source and software audit: zero training updates, no paid campaign,
no new toy qualification or default selection. Read
[EXPERIMENTATION.md](../../../EXPERIMENTATION.md), the
[compiled memory](../EXPERIMENT_MEMORY.md), and the
[existing current leaderboard](../technique-inventory.md) before acting on the
recommendations. That leaderboard remains the one current board for its goal.

## What Hyperchamber contributed

The inspected revision is
[`f9b92f55`](https://github.com/HyperGAN/hyperchamber/blob/f9b92f5518f1b23e6873e7da63bc31aac3819aae/hyperchamber/selector.py).
Its selector treats a scalar as a constant and a list as choices. Values can
be strings, dictionaries, lists or callable objects; the selector does not
interpret an optimizer or a loss. A mixed-radix index selects a combination
without expanding the Cartesian product. It supports indexed/serial access,
random access, saved configurations, and sorting recorded `(config, result)`
pairs by a caller's function.

HyperGAN's inspected
[2017 random-search source](https://github.com/HyperGAN/HyperGAN/blob/79f073a068f2d2c2ad9327ce0eb4d12f4f62a27c/hypergan/search/random_search.py)
builds nested trainer/model/loss dictionaries through those selectors. Its
trainer search chooses among six TensorFlow optimizer classes and independently
draws role settings, then explicitly assigns the D optimizer class to the G
class. That correlation is application logic, rather than an optimizer-aware
Hyperchamber feature. Nested sections are sampled when constructed; the outer
selector receives the already sampled dictionaries.

The useful ideas are whole-dictionary choices, cheap indexing of a large finite
space, and separating configuration generation from execution. Avoid copying
these implementation details:

- `randint(0, count_configs())` includes an extra index. That index wraps to
  configuration zero, giving zero twice the probability of other combinations.
- Random draws use Python's process-global RNG, allow duplicates and have no
  saved search-stream state. Serial access also wraps past the end.
- `Selector(initialStore={})` has a mutable default; the module also exposes a
  global selector. Neither is a good source of isolated study state.
- Callable serialization and
  [dynamic imports](https://github.com/HyperGAN/hyperchamber/blob/f9b92f5518f1b23e6873e7da63bc31aac3819aae/hyperchamber/__init__.py)
  do not establish validated public capability bindings.
- Result sorting supplies no fixed tasks, numerical gates, reservation policy,
  comparable source cohorts, or independent confirmation.

## Forge machinery on the audited base

The authoritative path is
[`configuration_search.py`](../../../experiments/forge/configuration_search.py),
with field ownership in
[`boundaries.py`](../../../experiments/forge/boundaries.py), mechanism checks in
[`techniques.py`](../../../experiments/forge/techniques.py), and public factories in
[`recipes.py`](../../../particlegan/recipes.py).

| Need | Current behavior | Assessment |
| --- | --- | --- |
| Assign values from finite choices | Schema-v2 `grid`, Cartesian product, maximum 256 configurations | Already supported |
| Keep related choices together | Named axis of dictionaries, all choices binding the same fields | Already supported |
| Avoid an unwanted product | Finite union of grids, rejecting duplicate choices | Already supported |
| Choose literal pairs | `betas: [[b1,b2], ...]`; each pair is one choice | Already supported |
| Draw a bounded random subset | No random strategy or distribution fields; all declared combinations expand | Missing convenience layer |
| Search optimizer/loss categories | `optimizer_family` and `loss` are technique-owned and forbidden grid axes | Intentional boundary; use structural candidates |
| Search optimizer numbers | LR, D/prior multipliers, moments and shared epsilon are whitelisted | Supported within mechanism boundaries |
| Separate G/D moments and epsilon | One network `betas` and one `eps`; `prior_betas` is a different role | Public Recipe gap |
| Validate meaningful axes | Reject fields inactive or task-owned on every tuning task | Retain; especially useful for conditional choices |
| Reproduce and resume | Freeze declarations/source/runtime, content identities, queue reuse and checkpoint contracts | Stronger than the historical selector |
| Select a whole recipe | Required PASS counts by ascending tier, then content hash; all trials must be terminal | Retain; no per-task winner mixing |

For example, this existing grid syntax gives two coupled recipes:

```json
{"rates": [
  {"lr": 0.001, "d_lr_mult": 1.0},
  {"lr": 0.002, "d_lr_mult": 0.5}
]}
```

This is the `grid` portion of a complete search, not an executable study by
itself. The study still needs its hypothesis, base candidate, family, protocol,
view, runtime, tier cap and campaign budgets.

The whitelist is deliberately narrower than everything numerical in Recipe.
For example, changing beta1 from zero to positive activates a moment mechanism;
zeroing a penalty, adding a schedule or toggling AMSGrad can likewise require a
structural candidate. A categorical front end should respect these checks.
Existing loss comparisons already use separate structural bases and numerical
searches, as in
[the least-squares search](../../../configs/forge/searches/pure-bcap-least-squares-joint-rates-v2.json).
The displayed Pure BCAP family groups selectable losses; its family label does
not authorize crossing the stricter technique signature inside one grid.

The current optimizer choices are `adam` and `formulation` (Adam with the
declared K3P/KA2 interventions). SGD, RMSProp and the other historical classes
are not public Recipe alternatives today; supporting them would require
shared API, adapter and checkpoint work before categorical admission.

Search planning checks full candidate and aggregate campaign reservations.
Admission freezes every request before submitting any of them. New ordinary
requests finish runnable independent peers in the current tier; required
failures then block higher tiers. Historical requests retain their own policy.
Missing capabilities remain blockers, and exact compatible evidence can be
reused. The separate
[policy-family search](../../../docs/forge-policy-family-search.md) has its own
host adaptations, serving law and persistence gates; it is not interchangeable
with ordinary clean-MoG search.

Selection currently ranks gate counts, not continuous quality or convergence
speed. A hash tie-break is deterministic presentation, not scientific evidence
that one tied configuration is better. The
[convergence-selection plan](../../../docs/forge-convergence-selection-plan.md)
contains proposed timing work and historical three-task/fail-fast examples;
use current declarations for execution. At this base, `discriminator_stability`
revision 5 has six required Tier 1 tasks plus one separate clock diagnostic.

## Can we represent Halloween's optimizer and loss?

The inspected file is `/mnt/ml7tb/dev/small_linear/halloween.json`, 5,158 bytes,
SHA-256 `158eefe34b89eace25d206a8dc4ee8d5990f2658f396f632984832cd122ef223`.
[The exact extracted trainer/loss](halloween-optimizer-loss.json) is portable;
the architecture and auxiliary `eloss` are outside this requested extraction.
The original executed HyperGAN revision, TensorFlow version and training
receipt are unavailable. Its filesystem mtime is 2017-08-21; that does not bind
an execution recipe or prove the configuration won a scored search.

| Setting | Halloween value | Current representation |
| --- | --- | --- |
| G optimizer | TensorFlow `AdamOptimizer` | Native PyTorch Adam is available; update-law translation still required |
| D optimizer | TensorFlow `AdamOptimizer` | Same qualification |
| G LR | `0.008020980209802098` | `lr` |
| D LR | `0.003947939479394794` | `d_lr_mult = 0.4922016232592352` |
| G betas | `(0.6119661196611966, 0.5612256122561226)` | Shared `betas` can set G, but also changes D |
| D betas | `(0.1051710517105171, 0.7203172031720317)` | No D-specific Recipe field |
| G epsilon | `0.3087330873308733` | Shared `eps` can set G, but also changes D |
| D epsilon | `0.007500075000750007` | No D-specific Recipe field |
| Loss | Least squares, fake/real/G labels `[-1,1,1]` | `loss="least_squares"` instead uses `[0,1,1]` |
| Decay declaration | Exponential, rate `.96`, steps `50000` | Not the current Recipe cosine/constant schedule |

The public role factories **do** accept separate Adam keyword overrides:

```python
from particlegan import get_recipe

recipe = get_recipe("bcap", lr=0.008020980209802098,
                    d_lr_mult=0.4922016232592352)
opt_g = recipe.make_generator_optimizer(
    generator.parameters(), betas=(0.6119661196611966, 0.5612256122561226),
    eps=0.3087330873308733)
opt_d = recipe.make_critic_optimizer(
    discriminator, betas=(0.1051710517105171, 0.7203172031720317),
    eps=0.007500075000750007)
```

This illustrates optimizer-field expressiveness only. It is not a Halloween
reproduction or a Forge recipe: `bcap` also declares its own critic penalty,
the snippet uses PyTorch epsilon semantics, and the requested loss is absent.
Forge's trainer path consumes the declared Recipe and does not expose these
separate role overrides as search fields. Hidden factory calls would bypass
the recipe's scientific identity. `prior_betas` cannot substitute for D betas.

### Separate consumed settings from historical baggage

In the inspected
[TensorFlow-era trainer](https://github.com/HyperGAN/HyperGAN/blob/79f073a068f2d2c2ad9327ce0eb4d12f4f62a27c/hypergan/trainers/base_trainer.py),
optimizer kwargs are filtered against the selected constructor. Adam consumes
the role beta1/beta2/epsilon values, with LR passed separately. Momentum, rho,
role decay, accumulator initialization and the floating `g_global_step` /
`d_global_step` fields are not Adam constructor settings. Preserve these in
the extraction as provenance; do not search them as if they affect Adam.

In the inspected
[least-squares implementation](https://github.com/HyperGAN/HyperGAN/blob/79f073a068f2d2c2ad9327ce0eb4d12f4f62a27c/hypergan/losses/least_squares_loss.py),
the labels define the objective. Its alpha/beta/gamma, initial-k/k-lambda,
label-smooth, reverse, type and use-k fields are not consumed there. Minibatch
is disabled in this file. The associated BaseLoss reduces by mean, despite
the saved `reduce_sum` string. These are reference-snapshot findings, not a
claim that the missing original runtime necessarily used this implementation.

Schedule provenance is particularly unresolved. The inspected
[August revert](https://github.com/HyperGAN/HyperGAN/blob/55c169c9f19046f79cada264e9c5f9e3207da410/hypergan/trainers/base_trainer.py)
does not consume the decay declaration; the other inspected snapshot does,
and advances its shared global step through optimizer applications. Its commit
was applied in October. Do not infer the original decay clock from the JSON.
The inspected AlphaGAN also composes multiple losses/trainer roles; extracting
these two sections does not reproduce its full training system.

### Two numerical differences that prevent a faithful direct import

For fake/real/G labels `(a,b,c)`, the inspected historical objective is

```text
L_D = 0.5 mean((D_real-b)^2 + (D_fake-a)^2)
L_G = 0.5 mean((D_fake-c)^2)
```

Forge fixes `a=0, b=1, c=1`; Halloween sets `a=-1`. This changes the fake
critic gradient. The explicit software fixture gives D loss `2/3` in Forge
versus `7/6` for Halloween, a value gap `.5` and maximum fake-score gradient
gap `1/3`. G's scalar objective agrees on this fixture. There is no public
label override; changing labels needs a declared loss formulation. Changing
critic output parameterization to compensate would also change the fixed task.

Legacy
[TensorFlow Adam](https://github.com/tensorflow/tensorflow/blob/9e76bf324f6bac63137a02bb6e6ec9120703ea9b/tensorflow/python/training/adam.py)
uses epsilon with the uncorrected second moment; native PyTorch Adam adds
epsilon after second-moment bias correction. For constant beta2 and dense
gradients, equality requires

```text
eps_PyTorch(t) = eps_TensorFlow / sqrt(1 - beta2^t)
```

No single constant PyTorch epsilon reproduces every step. At G's first step,
the equivalent epsilon is about `.466082`, not `.308733`; at D's it is
`.0141818`, not `.00750008`. For a `.001` gradient and zero initial moments,
G's analytical first displacement is `1.71725e-5` under legacy TF versus
`2.58964e-5` using the same numeric PyTorch epsilon. This is an analytic dense
update comparison, not an executed TensorFlow training result. The source
does not bind the unavailable original TensorFlow installation.

## Recommendations, in implementation order

1. **Add explicit role settings to the public Recipe.** Candidate fields such
   as `d_betas` and `d_eps` should inherit shared values when omitted; define
   prior/G/E precedence explicitly. Propagate them through all public factories,
   schedules, task adapters, validation, ownership, technique signatures,
   frozen recipe hashes and checkpoints. Preserve archived identities through
   an explicit compatibility/version path. Exercise genuinely different role
   values so an ignored-field bug cannot pass unnoticed.
2. **Declare the historical loss/update variant.** Add explicit least-squares
   labels or a registered named loss variant, including the correct joint
   encoder stream. For exact dense TensorFlow-Adam semantics, expose a named
   update variant with saved role step clocks, moments and bias-correction
   state. Alternatively declare a PyTorch adaptation and retain its measured
   differences. Bind any decay decision to its actual clock and source. Keep
   task-owned auxiliary objectives, prior settings and architecture explicit.
3. **Add a small bounded search compiler inside Forge.** Use explicit tagged
   choices so a literal list is distinguishable from a choice set. Sample
   finite configurations without replacement with an isolated search RNG,
   persist its derivation/state and the resolved draw manifest before training,
   and allow indexed mixed-radix access for large spaces. Emit at most 256
   reviewed unique configurations through existing admission checks. A future
   log-scale distribution should compile to a frozen finite manifest rather
   than generate new trials during training or recovery. This needs no new
   external library.
4. **Represent categories as a declared candidate roster.** Optimizer/loss
   choices should resolve through an explicit capability registry to structural
   candidates, then separate numerical searches within each signature. Share
   one immutable campaign/round budget, bind matched tasks and source/runtime,
   and retain whole-configuration selection. Conditional optimizer options
   belong only to their effective category; do not pad every choice with inert
   settings. Existing idea/search/inventory machinery supplies the execution
   pieces; the compiler would simplify their preparation and common manifest.

The immediate research target should be a clearly named **Halloween-inspired
optimizer/loss transfer**, after these representational decisions. Choose
whether it tests the raw historical labels/update rule or a documented PyTorch
adaptation. Keep one global trainer configuration across the unchanged tasks,
and declare the complete trainer delta. If LR, moments, epsilon and loss all
change together, that tests the bundle; it does not identify which factor helped.
For causal follow-ups, freeze finite factor groups and stopping rules first.

The existing
[Pure BCAP readout](../pure-bcap/README.md) already records least squares at
`.00425` with **2/6** required Tier 1 passes and at `.0010625` with **1/6**.
Both fail Gaussian, ring and word acquisition; neither is qualified. Those
runs use different moments, epsilon and labels from Halloween, so they neither
validate nor rule out its bundle. Preserve their original source receipts and
the selected whole family row; do not rerun them unchanged or assemble a row
from their successful cells.

For a future ordinary study, use protocol seed **0**, the public deterministic
initializer and isolated/checkpointed constructor, data, training-noise and
evaluation streams. Keep each task's architecture, target/data law, seen batch
sequence, prior, sampling, update budget and evaluation cadence fixed across
candidates. Keep explicit identity/zero fixtures in separate cohorts. Bind all
trained gates to their actual recipe and clean/noisy serving cohort.

Budget from the actual planned jobs: at this base the six required Tier 1
tasks reserve **2,220 seconds per candidate**; including the separate
300-second clock diagnostic gives **2,520**. Freeze both candidate and aggregate
campaign caps before enqueue, finish runnable peers, and stop higher-tier
progression on required failure. A new public-API toy qualification needs its
declared numerical metric and actual-training goal GIF. Screening remains
provisional; independent confirmation, accepted calibration and registered
robustness are still required for public-default adoption. No seed-only study
follows from this audit.

## Reproduce the audit

[audit.py](audit.py) uses the current public optimizer/loss factories and Forge
grid/signature validation on explicit constant software fixtures. It constructs
no learned toy, takes no optimizer steps and touches no operational queue.
[audit.json](audit.json) records expected rejections, exact numerical gaps,
runtime and source hashes. [external-sources.json](external-sources.json)
pins inspected external source files and their byte hashes; these are reference
snapshots, not inferred historical execution receipts.

From a checkout of the original audit commit `f3100080` (the script retains its
original rejection checks):

```sh
mkdir -p runs/forge/hypergan-search-audit
python reports/forge/hypergan-search-audit/audit.py \
  --output runs/forge/hypergan-search-audit/audit.json \
  > runs/forge/hypergan-search-audit/audit.log 2>&1
tail -F runs/forge/hypergan-search-audit/audit.log
```

Optionally add `--source /mnt/ml7tb/dev/small_linear/halloween.json` to verify
the complete original file hash and its extracted sections. The extracted
fixture allows reproduction without that mount. The five relevant existing
test modules (`test_forge_configuration_search`, `test_forge_boundaries`,
`test_forge_techniques`, `test_adversarial_losses`, `test_pure_bcap_recipe`)
passed: **169 tests in 5.98 seconds**. Raw pytest output remains local at
`runs/forge/hypergan-search-audit/pytest.log` and is easy to tail. Commit only
this report, compact receipts, extracted input and reproduction source.
`python -m experiments.forge validate`, local report-link checks, portable
receipt reproduction and `git diff --check` also passed.
