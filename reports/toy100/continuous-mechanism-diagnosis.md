# Constant-rate instability: output motion and passing-state restarts

The failing constant-rate recipe moves generated points farther in one G
update than the mode-hold quality radius. Most of that motion comes from
the generator network. This is directly observed output amplification;
neither these measurements nor a passing checkpoint establish a true game
equilibrium or a rotational limit cycle.

The exact simpler recipe's scheduled and constant controls reproduce the
original 1,200-update frozen host, seed 0, all 24 live observations and original
thresholds. The observer adds no training RNG draws. Metrics below average
observed updates 1000–1200, sampled every ten updates. Output differences use
the same complete finite prior support without output noise; G is applied
first in the attribution and prior movement second.

| Run | G output motion RMS/update | Prior output motion | Combined motion | Cap active fraction |
| --- | ---: | ---: | ---: | ---: |
| Scheduled .00425 | .01971 | .00204 | .02153 | .00130 |
| Constant .00425 | .40625 | .03719 | .44011 | .23270 |
| Constant .0025 | .31984 | .02266 | .33953 | .21336 |

The HQ radius is .21. The claim that the cap is wholly inactive in the
failing trajectory is false: about 23% of its observed sample gradients
exceed the cap. The median bias-corrected Adam denominator is .000791 for
G, .001826 for D, and .05196 for prior coordinates. Thus a single common
denominator floor would have very different effects on the three roles.

Four diagnostic branches share the exact scheduled prefix through update
1000 (8 modes, HQ .999756), then restore original constant group rates.
Freezing one role is an intervention for attribution, not a proposed recipe.

| Constant-rate restart | HQ at 1050 / 1100 / 1150 / 1200 | Final modes |
| --- | --- | ---: |
| All roles learning | .2505 / .2781 / .5283 / .5046 | 6 |
| G network frozen; prior and D learning | 1 / 1 / .9565 / .9976 | 8 |
| Prior frozen | .4084 / .8186 / .3403 / .3130 | 7 |
| D frozen | .6970 / .7227 / .6428 / .8279 | 8 |

This isolates generator-network motion as a dominant local destabilizer.
The host has twelve equally sampled particles for eight target modes and
output noise .029 versus target width .07, so exact distribution matching is
unavailable. Irreducible representation mismatch can create persistent game
forces; a passing state should not be called an equilibrium without measuring
the joint field.

## Completed bounded mechanisms

Nine constant-rate R1+R2 rows crossed LR {.001,.0025,.00425} and coefficient
{.01,.1,1}. One passed mode-hold: LR .001, coefficient .1, terminal passing
suffix 14. An uninstrumented production replay and independent episode
regrade confirmed that pass. Production trajectory then failed at MSE .261012.
A separate uninterrupted ring continuation failed 92/120 dense observations
after update1200, including zero modes at1400. This is not a shared or
continuous-learning solution. Following the user's subsequent direction,
zero-centered regularizer exploration stopped. A data-scaled variant failed
during archive setup before any training and was not retried.

Three fixed output-displacement bounds applied to the actual G Adam proposal
used radii .0145, .029 and .058, all at constant LR .00425. Backtracking
interpolates G parameters on the same clean training inputs, retains ordinary
Adam moments and the full prior update, and has no training-age dependence.
All failed the original mode-hold gate, finishing at 4/.4734, 3/.5671 and
7/.8440 modes/HQ. The ordinary control reproduced 7/.369385 exactly.
Independent finite-large-radius identity checks verified exact optimizer,
RNG, noise and metric parity. A separate .058-bound run continued to 4,800
updates and still failed sustained dense quality; only 102/360 later checks
passed and the final endpoint had 7 modes. Therefore slower acquisition alone
does not explain this bound's failure.

## Research direction after isolation

[C-CHAIN (2025)](https://arxiv.org/html/2506.00592v1) connects uncontrolled
output changes to plasticity in continual reinforcement learning. It motivates
measuring function changes here, but supplies no GAN convergence guarantee.
[The 2026 asymmetric-GAN paper](https://arxiv.org/html/2601.13920v1) analyzes
zero-centered curvature under smoothness and identifiability assumptions;
neither its method nor those assumptions establish this host's stability.

The next completed game-update comparison used a fixed Adam metric during
both extragradient evaluations and compares identical versus independent
samples. It advances moments once, uses two joint gradients, and takes its
final step from original parameters. This preserves zero game fields and
differs from the previous ExtraAdam experiment, which both redrew samples and
advanced moments twice. [Same-sample extragradient analyses](https://proceedings.mlr.press/v108/mishchenko20a.html)
motivate testing sample reuse, while
[the August 2026 comparison](https://arxiv.org/abs/2608.06182) reinforces that
sampling variants require different assumptions; neither automatically
guarantees convergence for this nonlinear Adam-preconditioned game.

The matched warm comparison retained byte-identical models, Adam moments,
EMA and RNG at update1000. Scheduled Adam passed 200/200 subsequent live
checks, ordinary constant Adam passed6/200, and both fixed-metric EG variants
passed0/200. Same-sample replay was verified at all200 updates. At update1055,
the same-sample prior correction reached max |g2/denominator|=3.30e11 and
joint clean output movement8.26e17; Adam second moments subsequently
overflowed. Independent-sample EG failed similarly at1038. The initial
receipt failed JSON serialization on infinity; the preserved rerun recorded
nonfinite diagnostic paths explicitly and confirmed actual numerical failure.

Reversible same-sample EG then tested the local secant condition
`||Ftrial-Fbase||_P <= c ||Fbase||_P`, with
`P=diag(group_lr/first-gradient_Adam_denominator)`. Rejected joint trials
halved alpha; the next update tried twice its last accepted alpha, capped at1.
Moments advanced once, every trial replayed the same data/noise, and accepted
parameters came from the original point. All three c values passed200/200
warm checks. Independent bilinear and actual-host tests verified the method,
RNG replay and moment/query counts.

| Secant c | Warm mean alpha | Warm max clean output RMS | Cold trajectory MSE | Cold passing checks |
| --- | ---: | ---: | ---: | ---: |
| .25 | .11156 | .01852 | .28720 | 0/24 |
| .50 | .26719 | .02475 | .35397 | 0/24 |
| .90 | .31500 | .06585 | .13317 | 0/24 |

Every cold row failed the original trajectory threshold .02. Fail-fast
ordering stopped them before cold mode-hold or long continuation. This
separates passing-state stability from acquisition: scalar step control can
stabilize the former while failing the latter. No row is a winning
continuous-learning recipe. Source bytes and transforms are archived under
the fixed-metric and secant artifact directories.

A bounded matrix-free implicit response then solved
`(I + alpha sqrt(P) J_F sqrt(P)) u = -alpha sqrt(P) Fbase`, with
`delta = sqrt(P) u`. It used at most8 GMRES directions, same-sample
finite-difference Jacobian products, linear relative residual<=.1, a
proposed-correction norm bound, and an independently evaluated actual
nonlinear implicit residual<=.5. Parameter-dtype rounding was measured.
All nominal role LRs remained constant; rejected solves halved alpha.

The warm branch passed200/200 checks (minimum HQ .98315), using2,273 gradient
evaluations per player for200 updates and2073 verified same-RNG queries.
Accepted alpha min/mean/max was .0625/.2975/1; maximum linear/nonlinear
residuals were .09964/.49524. Four independent tests covered exact stiff
bilinear response, zero-field no-op, original-host identity, and active
RNG/query/moment accounting. Sitting at a matched target is acceptable;
these tests impose no minimum-motion requirement.

Cold trajectory still failed: MSE .253794,0/24 passing checks,400 moment
updates and3,237 gradient evaluations per player. Mean alpha was .03433,
falling to .00773 over the last50 updates. Of405 rejected solves,395 failed
the nonlinear residual,6 exhausted the linear residual target, and4 exceeded
the proposed-correction bound. Thus more GMRES work alone is not supported
as the next fix. Cold mode-hold and shift tests were skipped after this
acquisition failure. The next bounded comparison used cross-only
competitive response: the [CGD authors](https://f-t-s.github.io/projects/cgd/)
explicitly distinguish it from the full-Jacobian Newton response used here.

That comparison retained the same metric and solver limits but included
only cross-player Jacobian blocks, treating G and its prior as one player.
Two separated finite-difference perturbations recover those blocks despite
the host's detach and temporary requires-grad flags. The nonlinear residual
also uses each player's own base parameters and the opponent's proposed
parameters; a full joint residual is recorded as an observation, not used
for acceptance. A quadratic/bilinear analytic case confirms that own
curvature is omitted.

Cross-only response failed warm stability at196/200 checks (minimum HQ
.88940), despite final8 modes/HQ .98853. Mean alpha was .75938 and cost was
2,729 gradient evaluations per player for200 updates. Cold trajectory ended
at MSE .020058 versus the unchanged .02 threshold, with0/24 passing checks;
it is still a failure, with no qualifying suffix. It used5,846 evaluations
for400 moment updates. This is much better acquisition than full-Jacobian
response, but does not solve sustained stability or shared acquisition.

The cold cross-only run was mistakenly launched in the same tool batch
before consuming the warm failure verdict. It completed before cancellation
could take effect. Its evidence is retained as an unplanned diagnostic,
with an explicit protocol-deviation receipt, and has no eligibility credit.
No cold mode-hold, extended hold or shift run followed that failed row.

A fresh torch 2.14 fork does not replay updates 1194–1197. Stock cross-only
fails at update 1002 instead, with generator output RMS .203. Own/cross ratios
there stay near the other warm steps and below archived acquisition ratios.
An optional matched-support output cap of .05 then passes the fresh warm
window and still fails cold trajectory at MSE .06901. See
[cross-drift-replay.md](cross-drift-replay.md).

For a cheap next investigation, replay the warm failures at updates1194–1197
and compare their functional movement with successful acquisition updates.
All four failed warm updates used alpha1. Their accepted cross residuals were
.223/.177/.320/.132; full-joint residuals were .581/.571/.478/.474. Simply
adding a full-joint residual<=.5 guard would miss two of those failures while
rejecting133/200 warm and305/400 cold accepted proposals. The existing
receipts therefore do not support another residual-threshold sweep. The
unresolved problem is retaining useful opponent response during acquisition
while avoiding the later functional drift; longer hold and matched shift
versus frozen-control tests remain required for any future survivor.

Exact values and raw artifact paths are in
[the compact diagnosis](continuous-mechanism-diagnosis.json). Scratch scripts
are `continuous_mechanism.py`, `output_trust_scratch.py`,
`output_trust_probe.py`, `fixed_metric_extra_scratch.py`,
`secant_extra_scratch.py`, and `implicit_extra_scratch.py`. No production
implementation, seed, model, or original quality threshold changed.
