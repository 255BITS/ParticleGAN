# Stable functional fitting still needs memory of observed support

No production replacement for LR decay is qualified. The current pre-start
anchor candidate passes the original ring acquisition gate, two forms of
continued training, and same-target model-error recovery. A separate
one-update counterexample nevertheless removes a mode when the next
minibatch temporarily omits it. Longer training is withheld for that
memoryless rule. This is a specific, potentially repairable mechanism,
not a proof that constant-rate learning is impossible.

| Fixed candidate / gate | Result |
| --- | --- |
| Pre-start anchor saved failure windows | 44/44, eight modes/HQ1 |
| Borrowed passing-state warm | 200/200, eight modes/HQ1 |
| Borrowed passing-state later hold | 1200/1200, eight modes/HQ1 |
| Original cold trajectory400 | PASS, identity MSE .000942662 |
| Original cold ring1200 | PASS, all24 observations eight modes/HQ1 |
| Continue its own acquired state1201–2400 | PASS,1200/1200 |
| Same-target +.35 output-bias error at2400 | Recovers eight modes/HQ1 on the first update; final5 pass |
| Paired unperturbed / frozen perturbed controls | 50/50 pass /50/50 fail |
| Next D bank conditioned to omit component0 | **FAIL:8→7 modes in one actual neural update**, HQ1 |

The [source-bound acquisition and borrowed-state archives](continuous-evidence/round6-sample-anchor-qualified/)
and [own-state, response and missing-bank archives](continuous-evidence/round6-anchor-own-and-missing/)
include exact sources, raw results and full state snapshots. All training
keeps G/D LR .00425, prior LR .0085, the original noise horizon, and one
native Adam update per player. No seed or coefficient sweep was used.
The original scheduled/disabled controls and the exact first200 training
state pass the [explicit evaluation-counter reconciliation](allocation-prefix-parity.md).

## What the fitting change establishes

The old target fit started from the native GAN proposal. In a linear output
map, its minimum-norm correction leaves the native move's Jacobian-kernel
component intact. The [one-state experiment](anchor-parameter-motion.md)
measures that effect. Starting the same target fit from pre-G instead nearly
eliminates the unnecessary component while keeping the output target and
numerical budget unchanged.

The [independent long-hold assessment](anchor-internal-state-assessment.md)
finds post-native G norm15.75→19.41, versus pre-start13.26→13.23. Median
first-fit Jacobian conditioning grows8.98→66.71 in the old rule, versus
8.78→9.06 with pre-start fitting. These are finite measurements. The same
assessment gives an exact nonlinear counterexample where minimum-motion
updates accumulate parameter drift around bounded output loops; no general
parameter-boundedness theorem is claimed.

Every accepted update in these tests selects the fitted data objective.
The method adds a distinct-group coverage objective and is not an
objective-preserving repair of the original GAN optimizer. Its
[conditional output-invariance argument](anchor-invariant-region.md)
requires correct bounded-error group estimates. The next test deliberately
checks that premise instead of relying on a longer lucky draw sequence.

## Absence from a minibatch is not evidence of disappearance

Under the unchanged uniform eight-component sampler, the event that a
128-row bank contains no component0 has probability `(7/8)^128`, about
3.776e-8. Sampling from the seven remaining components is exactly the
conditional law of that bank. The complete target distribution, G's real
bank, and evaluation means remain unchanged. This is neither a target shift
nor a new training seed.

The [free-output diagnostic](anchor_missing_batch_group.py) starts at the
completed borrowed-state hold. The ordinary next bank produces eight
inferred groups and retains eight-mode quality. The conditioned bank gives
seven inferred groups; the current-bank anchor objective sends the formerly
covered component0 particle to another observed group, reaching seven modes
while reducing its objective from .441896 to essentially zero.

The [actual neural diagnostic](anchor_missing_batch_neural.py) independently
starts at the candidate's **own acquired** update2400 state. It conditions
only D's real bank and replays it identically in all three native phases.
The normal arm stays at eight modes; the conditioned arm selects a converged
joint fit and loses component0. Both end with identical training RNG states.
Thus this is an actual one-update failure, beyond the free-output example.

The candidate also passes a real model-error response test: shifting G and
its matching EMA by +.35 in x is corrected immediately, whereas the frozen
shifted model fails all50 checks. Responsiveness does not repair the
missing-information semantics of the current-bank objective.

The next cheap filter retains data-derived group sufficient statistics,
updates observed groups, and preserves absent groups. New unmatched observed
groups can still be acquired. Matching and discovery need explicit separation
conditions; the memory must be serialized as learner state for every resume.
A growing data estimate must not reduce the ability to make a full correction
when G develops an error. No longer neural run is authorized by a tiny memory
test alone: the omission/discovery filter, exact replay, warm/hold and cold
gates must all pass for the new source-bound rule.

## Limits beyond this ring

The [production first-bank check](sample-anchor-production-geometry.md)
finds all100 true groups in each of the actual fixed2048-row banks, and the
MST correctly separates all100. However, anchor loss zero permits arbitrary
duplicate counts and does not enforce the production mass-fidelity gate.
The production generator is affine, so a structured output fit may avoid
the current dense-Jacobian scaling cost; that implementation is not tested.
Within-group shape and conditional fidelity remain separate requirements.

The [latest research review](sample-anchor-fidelity-research.md) motivated an
analytic noise-aware MMD test. That [one-width cheap filter](sample-anchor-mmd-filter.md)
fails: the cold acquisition target worsens its MMD, and warm MMD descent
loses coverage. It receives no1200-step or neural run. No result here implies
that all proper distribution objectives or all constant-rate methods fail.
