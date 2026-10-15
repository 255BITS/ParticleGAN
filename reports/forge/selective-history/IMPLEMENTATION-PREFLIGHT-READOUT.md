# Selective generator history: implementation and admission

**Status: implemented for software review; all four studies remain DRAFT and
the saved-state preflight is BLOCKED. No new scientific training outcomes,
qualification, or default change are claimed.** Required original
two-pole/Ring/conditional windows are not yet fully available, and strict
next-update checks remain incomplete at some exhausted endpoints.
Available-state checks cannot replace missing windows.

The [frozen preflight](PREFLIGHT.md) defines the evidence and acceptance tests.
The [exact draft roster](stage-roster.json) binds the source digest, runtime,
candidate revisions, declaration hashes, two original tasks and job identities.
It is a reviewed planning record, not a queue reservation or executable approval.

Completed [available-state probes](available-state-readout.json) support a
limited mechanism result: S changed the actual normalized generator displacement
on all 31 noninitial panels in each of four recovered windows, with no registered
first-order reversal of useful transport. Across 128 real and 32 synthetic
panels, the three arms T/W/S produced 480 finite proposals in 13.10 seconds.
Each proposal started from restored model state; no training trajectory continued.

Finite transport loss still increased over the same panel's starting value:

| Original window | T | W | S |
|---|---:|---:|---:|
| Gaussian step 1000 | 18/32 | 17/32 | 19/32 |
| Shifted Gaussian step 4000 | 0/32 | 0/32 | 0/32 |
| Gaussian step 6000 | 18/32 | 10/32 | 18/32 |
| Unequal-mass step 1200 | 14/32 | 10/32 | 14/32 |

These descriptive counts use strict numerical increases, without a tolerance.
They establish neither broad finite-step stability nor superiority for S. The
synthetic control supplies no missing-window credit. Missing two-pole, failed
Ring and conditional identity windows, plus incomplete next-update checks at
the exhausted endpoints, keep the overall preflight **BLOCKED** and every study
**DRAFT**. The compact readout records hashes of the external raw evidence;
the frozen preflight and roster remain unchanged.

| Arm | Definition | Candidate |
|---|---|---|
| B | Current BCAP/DualNorm/direction-blend baseline | [Canonical card](../../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) |
| T | B plus global/local-v2 transport, weights 1/1 and 32 projections | [Transport card](../../../configs/forge/ideas/bcap-selective-history-transport-v1.json) |
| W | T plus whole-generator gradient history | [Whole card](../../../configs/forge/ideas/bcap-selective-history-whole-v1.json) |
| S | T plus adversarial-only generator gradient history | [Selective card](../../../configs/forge/ideas/bcap-selective-history-adversarial-v1.json) |

## Implementation and source identity

The recovered original source manifest is
`9bc9e6ae14ccd13455bf8805bd68b9e3affd6e09e112a06be7a97b368248d61f`;
its `gradient_signal.py` SHA-256 is
`b7fd16b80604e7b0982aa426f0723b95ffa1d0637375a8e933598f2fbe16fdeb`.
Its role-level estimator uses decay 7/8, effective sample count, unbiased
variance, a debiased nonnegative signal and denominator epsilon 1e-16.
The archived caller attenuated the normalized update through learning rates.

The new [component wrapper](../../../particlegan/optim/component_history.py)
instead applies that factor **before normalization**. W observes the total
generator gradient and multiplies it by rho. S observes the current original
adversarial gradient `a` and replaces total gradient `g` by
`g + (rho - 1) * a`. This is a new adaptation, not an inherited historical pass.
The [estimator](../../../particlegan/optim/gradient_signal.py) preserves the
recovered arithmetic and adds explicit-channel input and state validation.

Only effective generator-player slots are transformed, including direct
generated coordinates. Encoder and learned latent-prior gradients are excluded;
the existing joint direction blend can still couple their final displacements.
The [component adapter](../../../experiments/forge/behavior_adapters.py) retains
the original optimizer groups when disabled, explicitly labels active G/E
groups, and delegates one backward to one joint optimizer. Existing protected
normals remain raw and current. Missing adversarial decomposition or ambiguous
optimizer ownership rejects execution. Neither scalar attenuation nor
first-order protection is a finite-step convergence guarantee.

Active history stops after any optimizer-step exception and rejects further
backward calls, retries and checkpoint export. This is a fail-stop guard, not
rollback: the failed base step may already have changed model tensors. Rebuild
the full trainer and restore its last valid complete checkpoint before
continuing. Loading optimizer state alone can clear the guard but does not
restore model weights. The disabled path retains the original base behavior.

The branch starts from `4728b8b030237aa00ef2188a8782cea247991c32`.
Ten current task declarations update source pins and provenance only:
architecture, target/data/prior/sampling laws, budgets, cadence and numerical
gates are unchanged. Older receipts retain their original identities. New
source files and declarations must be included in the eventual commit; no
branch name alone substitutes for the roster's source digest.

## Admission, costs and checks

The [diagnostic view](../../../configs/forge/views/bcap-selective-history-pole-ring-diagnostic-v1.json)
contains two-pole and Ring as diagnostic Tier 1 assignments. Each of four draft
studies reserves at most 300 + 300 seconds; the shared campaign ceiling is
2,400 seconds. Both independent jobs finish per admitted arm, and S must pass
both complete gates before ordinary Tier 1 is considered. The full staged
science ceiling is 108,600 seconds, with a separate 1,800-second software/probe
allowance. Actual scientific spend and queue reservation so far are zero.

Forge's diagnostic marker enters the job compatibility hash. Read-only checks
confirm that all four DRAFT requests are rejected by admission and that these
jobs cannot silently fill ordinary qualification cells. The current diagnostic
screen does not replace the six ordinary Tier 1 tasks. Later continuations
require their own passing producer and complete checkpoint/RNG state.

Forge validation and four-arm planning pass their structural checks. The
[software tests](../../../tests/test_component_history.py) cover recovered
estimator parity, raw normals, role routing, finite writes, resume, malformed
state, failure recovery and host plumbing. The final runtime suite passed all
44 checks; four active GPU cases also passed exact full-trainer resume with
learned and frozen priors. These tests, original-only disposable restore checks,
and completed fixed-state gradient panels are separate from scientific results.
Exact source/runtime and original checkpoints must satisfy every gate in
[PREFLIGHT.md](PREFLIGHT.md) before any study is made ready.

The CUDA preparation environment uses Python 3.11.15, torch 2.11.0+cu128,
NumPy 2.4.4 and SciPy 1.17.1 on RTX A6000. Both devices passed tiny arithmetic
and native-SVD checks. Fifty-five registry-pinned GIFs omitted by the sparse
checkout were restored from exact tracked blobs and hash-verified; no historical
grade or receipt was rewritten. Memory reports STALE for the new declarations;
it was not recomputed into new qualification claims.

Detailed local evidence lives outside Git:
[admission and costs](../../../../combination-review-20261014/EXECUTION-ADMISSION.md),
[implementation checks](../../../../combination-review-20261014/EXECUTION-IMPLEMENTATION.md),
[source recovery](../../../../combination-review-20261014/EXECUTION-SOURCE-RECOVERY.json),
[missing originals](../../../../combination-review-20261014/EXECUTION-MISSING-ORIGINALS.json),
and [original-only restore probes](../../../../combination-review-20261014/execution-probes/original-restore-parity.json).
Those workspace links are not portable published artifacts; the repository
specification, source, tests and roster above remain self-contained.

Read-only review, from the repository root in the declared environment:

```sh
python -m experiments.forge validate
python -m experiments.forge plan bcap-selective-history-adversarial-v1 \
  --study bcap-selective-history-s-pole-ring-study-v1 --show-boundaries
```

The plan must continue to show DRAFT until the frozen preflight is complete.
Regenerate source/runtime bindings after any implementation change before
submission; do not edit generated requests to bypass admission.

## Reproduction sources and independent review

Byte-for-byte copies of the executed [original restore checker](probes/original_restore_parity.py),
[frozen gradient exporter](probes/original_gradient_panels.py), and
[current optimizer probe](probes/current_counterfactual_maps.py) are retained
for review. These scripts bind the recorded local archive and artifact layout;
execute them from the original external `execution-probes` directory, where
their verified inputs live, rather than writing tensor outputs into this report.
The [independent audit](independent-map-audit.json) recomputed all 480 finite
readouts exactly and verified that evaluation consumed no CPU/CUDA RNG.

The original frozen specification remains byte-identical. In its algebraic
S equation, the executed floating-point operation order is `g + (rho - 1) * a`,
as fixed in the implementation before any real-state outputs. At the saved
step-4000 state the original next batch already uses target mean 3; this is a
pre-shift parameter state with shifted inputs, not a late stationary mean-2
window. The missing late stationary window remains explicit. No acceptance
threshold or roster member was changed after observing results.
