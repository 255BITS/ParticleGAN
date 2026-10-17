# Routed E22 quality diagnosis at `03e6d705`

This is the historical diagnosis before the token-unit and represented-mass
corrections. See the [current qualification guide](e22_routed_game.md) for the
implementation, compatibility boundary and subsequent validation.

The requested structural objective is to repair the adversarial game.
Output RMSE/MSE belongs to external evaluation and logging; it must not drive
row evidence, proposal ranking, acceptance or optimizer updates. The earlier
recommendation to use output-MSE structural selection or protection is
withdrawn. The measurements below remain observations, not controller targets.

The current API-initialized run does **not** show row controls degrading quality:
full and movable-without-controls have identical trajectories and accept no
moves through 1,200 updates. Both finish at held-out RMSE **.005592**. The
reported full-control disadvantage belongs to the deliberately constructed
conformance fixture. That fixture does show a remaining disadvantage at the
longer horizon, so the investigation does not qualify full controls as better.
These output measurements alone do not establish whether the game was repaired.

Two formulation issues need attention: the example's pooled critic changes
effective regularization with token count, and routed DV12 geometry ignores
routing mass. The first materially changes the measured training trajectory.
The second has a concrete refinement-invariance counterexample, but was not
the immediate support-clipping cause of this fixture's six early moves.

All measurements use the same 128-token, 128-row, four-dimensional task, batch
size eight, fixed initializer role keys and matched batch/paired-noise/DV12
draw traces. There is no seed sweep, learning-rate search or held-out proposal
selection. The [receipt](e22_routed_debug_results.json) retains the harnesses,
source hashes, trajectories and counterexamples. These are synthetic accuracy
diagnostics on an A6000, not Sliders or music-quality measurements.

## What caused the reported difference

Vetoing commits while retaining full probe/proposal execution reproduces the
movable baseline's entire 40-update trajectory exactly, including its losses.
The accepted moves cause the difference; probe execution alone does not.

| Current conformance arm, 40 updates | Held-out RMSE | Maximum token L2 error |
| --- | ---: | ---: |
| Movable, controls off | .023096 | .069956 |
| Full, actual commits vetoed | .023096 | .069956 |
| Full, current half/quarter history transport | .027090 | .094223 |
| Full, unscaled parent history inherited | .026624 | .093690 |
| Full, old row histories retained | .025588 | .092273 |
| Movable, DV12 draws consumed but perturbations omitted | .040312 | .128413 |
| Full, DV12 draws consumed but perturbations omitted | .039504 | .127706 |

The alternative histories reduce only part of the gap and alter later
proposals. Removing perturbations reverses the ranking but worsens both
absolute results. Neither intervention is a qualified replacement law.

The six accepted moves occur at updates 5, 13, 32, 35, 37 and 39. Five are
duplicates; only update 32 is antisymmetric. Their immediate held-out RMSE
changes sum to approximately **-.000052**, much smaller than the final
**+.003994** full-minus-baseline difference. Most of the difference develops
in subsequent learning. Feature-improving moves at 32 and 35 slightly increase
raw output error, as the existing optional output guard was designed to check.
That guard certifies the immediate clean function, not later adversarial steps.
It is outside the requested game-driven structural objective and should remain
disabled for this integration.

The 17.29% endpoint disadvantage overstates the typical early-trajectory gap:

| RMSE measure | Movable | Full | Full relative increase |
| --- | ---: | ---: | ---: |
| Update 40 | .023096 | .027090 | 17.29% |
| Mean of updates 31–40 | .048493 | .049358 | 1.78% |
| Mean of updates 21–40 | .090888 | .091305 | .46% |
| Mean of all 40 updates | .122903 | .123103 | .16% |

Both runs oscillate strongly: full is better at update 39 and worse at 40.
Nevertheless, the longer current run also leaves full worse in this fixture:

| Current conformance checkpoint | Movable | Full |
| --- | ---: | ---: |
| 40 | .023096 | .027090 |
| 120 | .053929 | .023530 |
| 400 | .014662 | .013303 |
| 800 | .004607 | .006833 |
| 1,200 | .003421 | .004318 |
| Mean of ten checks at 1,110–1,200 | .004742 | .006255 |

This is a real remaining quality limitation on the fixture, alongside a
phase-sensitive early comparison. It is not evidence that every accepted move
is harmful or that the API-initialized row controller has been exercised.

## Pooled critic and penalty units

`TokenErrorCritic.forward` averages token features before producing one score
per context. For `D = mean_t f(x_t)`, each token's derivative includes `1/T`.
KA2 then divides its squared input-gradient norm by the whole context size
`T*C`. On an exactly replicated token, its real R1 term therefore scales as
**1/T²**, despite unchanged critic scores. At 128 tokens it is 16,384 times
weaker than at one token; moving from eight to 128 makes it 256 times weaker.
The relevant native implementation follows its documented per-input RMS
contract. The mismatch is in applying that contract to this pooled example.

A bounded diagnostic changes only real R1 to token-count-invariant units,
multiplying it by `T²`. It preserves critic scores, model initialization,
training batches and noise draws. Every measured update is inside KA2's
initial pure-A phase. Native and token-scaled *whole-context RMS* fake caps
remain inactive throughout these runs; that does not establish that all
individual token caps would remain inactive.

| Initialization / penalty units | Movable RMSE at 40 | Full at 40 | Movable at 120 | Full at 120 |
| --- | ---: | ---: | ---: | ---: |
| Conformance / original | .023096 | .027090 | .053929 | .023530 |
| Conformance / invariant real R1 diagnostic | .038254 | .032036 | .011421 | .010211 |
| API / original | .215200 | .215200 | .048345 | .048345 |
| API / invariant real R1 diagnostic | .166833 | .166833 | .020341 | .020341 |

Changing this normalization materially changes quality and reverses the
conformance row-control ranking at update 40. This establishes a causal
normalization effect, not a universal claim that the stronger term improves
every checkpoint. It also does not qualify the diagnostic multiplier as a
complete implementation: the later B caps and EMA-gradient proximity need
consistent units too.

An existing-API implementation has also been qualified in the diagnostic
harness. It keeps the original context grouping and critic scores for the
game, but presents tokens as samples only to the penalty:

```python
class TokenPenaltyView(nn.Module):
    def __init__(self, critic, tokens):
        super().__init__()
        self.critic, self.tokens = critic, tokens

    def forward(self, error):
        context_error = error.reshape(-1, self.tokens, error.shape[-1])
        context_score = self.critic(context_error)
        return context_score.repeat_interleave(self.tokens, dim=0)

view = TokenPenaltyView(p.D, real.shape[1])
penalty = p.penalty(view, real.flatten(0, 1), fake.flatten(0, 1))
```

The root critic remains a direct child of the view, so the existing paired
penalty supplies the matching EMA view. The critic's full contextual forward
is retained; this does not flatten tokens before its computation or count
tokens as additional independent row-evidence contexts.

Replicated one-, eight- and 128-token checks pass for pure A and post-800
blend with nonzero EMA proximity and active caps, including critic parameter
gradients. Lazy calls and their multiplier pass. CPU-deserialized checkpoint
recovery at update 60 reproduces the remaining 60 updates exactly for both
initializations, including weights, controller state and outputs.

The four matched CUDA runs finish at:

| Tested token-local penalty view, 120 updates | Movable | Full |
| --- | ---: | ---: |
| Conformance RMSE | .011421 | .010211 |
| Conformance maximum token L2 error | .037923 | .035446 |
| API RMSE | .020376 | .020376 |

Full conformance accepts 19 moves; API accepts none. Individual token fake
caps activate on ten API updates, explaining the small difference from the
R1-only diagnostic. This is a tested implementation of the stated token-local
unit convention. Its 120-update runs do not qualify longer-horizon quality
after the KA2 blend, which the separate fixed CPU phase checks exercise only
for mathematical and API consistency.

## Routed DV12 ignores represented mass

`E22Policy._prior_view` supplies the raw table without `candidate.log_mass`.
DV12 uses all rows equally for bandwidth and nearest-support clipping. As a
result, an inactive row can influence training perturbations, and replacing
that inactive row with an exact half-mass parent duplicate can change the
noisy training function while leaving the clean function unchanged.

The CPU counterexample keeps identical Gaussian draws and frozen bandwidth:
clean mixed-code difference is zero, support radius changes **.166667 →
.333333**, and perturbed codes differ by **.149426**. With nonzero tied
key/value queries, clean half-parent gradient error is below `2e-17`, while
current DV12 produces `.012623` half-parent gradient error and changes
untouched-row gradients. Updating the raw-table bandwidth adds another change.

For the six actual early fixture moves, substituting the new support table
while retaining old codes and bandwidth produces **exactly zero** output
difference on fit, guard and test contexts. The invariant failure is therefore
a separate demonstrated gap, not the established immediate mechanism behind
their measured deficit. Noise removal alone is not a repair.

A separate clean two-site Adam witness finds no additional tied-key/value
transport defect: gradients halve correctly, cloned rows remain identical,
and K3P matches plain Adam when its sparse damping is inactive. Default Adam
epsilon leaves a `2.64e-8` next-output difference; reducing epsilon removes it
to floating-point precision. Deleting a child with nonzero mass still makes
half-parent history an approximation, as already documented.

## Another integrator's structural-scoring diagnosis

The user supplied this independent integrator report; its fixture and
receipts have not been rerun here. Of 49 accepted splits, 11 improved learned
critic-feature error while increasing mean output error on the guard set.
The integrator allowed `max_context_harm=1e-4`, whereas `RoutedRows` defaults
to zero. Their deterministic-orthogonal, matched GPU diagnostics reported:

| Integrator configuration | Validation RMSE |
| --- | ---: |
| Fixed | .008881 |
| Output-MSE split criterion | .008965 |
| Movable | .009086 |
| Original full | .009148 |

Zero feature harm or strict output protection rejected every split and
reproduced movable exactly. Disabling encoder/router monitor restarts
reproduced original full exactly. Replay and optimizer-state checks passed.

This supports the feature/output mismatch already observed at updates 32
and 35 here. The distinction between metrics matters: `max_context_harm`
bounds **feature** error, so even its default zero does not mathematically
guarantee nonincreasing raw output MSE. All splits being rejected in their
fixture is an empirical result. The separate `output_error_guard` checks
raw output error for fast and averaged functions, after feature-based
proposal selection; it does not change deletion evidence or candidate ranking.

Output-based selection and output protection serve different purposes, but
both inject an output-accuracy objective into structural control. They are
outside the requested design. Eleven moves worsening output MSE does not,
by itself, establish that those moves harmed the game. Conversely, improving
learned-feature distance does not certify game repair.

In their reported fixture, full is .68% worse than movable, and output-based
selection is 1.33% better than movable but still .95% worse than fixed.
These matched results isolate structural scoring as a contributor to their
output trajectory; they do not establish a general benefit or a game repair.
Their restart ablation
rules out those restarts for that run. Their report does not test the token
penalty or DV12 invariance issues above, so those remain separate findings.

## Work to route to ParticleGAN

1. Promote the tested penalty view into the routed example with an explicit
   token/context unit contract and checkpoint compatibility. Turn its
   replicated-token A/blend/EMA, parameter-gradient, lazy-call and recovery
   witnesses into regression tests. Keep adversarial scores unchanged and
   preserve native independent E22 defaults. Recheck quality through the KA2
   blend; the tested 120-update continuation is not a longer-horizon claim.
2. Define routed DV12 geometry using represented mass/support, with a test
   that an inactive-child exact duplicate preserves the clean **and noisy**
   function under identical draws. Include bandwidth refresh and per-site
   gradients. Keep native independent-particle DV12 unchanged.
3. Investigate structural evidence against the actual training game. Current
   `_measure` uses clean squared learned-feature discrepancy; it is not the
   RpGAN/KA2 objective or a certified game-health statistic. Compare complete
   model candidates with matched per-site DV12 and paired-output noise, the
   same critic and regularization convention, and separate protected game
   contexts. Look for persistent support or gradient-conditioning problems
   that a structural move addresses. A lower instantaneous generator payoff
   alone can reflect exploiting a fixed critic and is insufficient evidence.
4. Qualify game-driven moves against fixed and movable banks on an
   API-initialized task that actually exercises distinct bank capacity.
   Record real/fake score behavior, penalty phase/contributions, generator
   and critic gradients, row support/usage and persistent game dynamics,
   alongside external output metrics. Preserve exact replay, optimizer
   history, the fit/guard/test boundary and native E22 behavior. Keep
   `output_error_guard=False`; no output metric should rank or veto moves.
   The present API task accepts no moves even at 1,200 updates, so its tie
   cannot establish structural benefit. A validated game-repair statistic
   remains work to define and test, not a mechanism supplied by this report.

The API initializer itself is being used. Its current full-network calls also
initialize nonconstant adapter matrices, so this example starts with nonzero
residual output. A benchmark intended to mimic initially neutral LoRA branches
must declare zero output factors deliberately before initializing the remaining
trainable projections. This is an architecture choice to document, not a reason
to bypass the public initializer.
