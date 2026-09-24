# Constant-rate continuous learning: gate audit and one D:G allocation test

No reviewed candidate qualifies to replace the shared 22-task LR schedule.
The one new fixed-rate, fresh-data D:G allocation arm failed the original
mode-hold quality requirements. Its exact runtime sources, controls and raw
receipts are under [adaptive-d-allocation](continuous-evidence/adaptive-d-allocation/).
The [compact result](continuous-evidence/adaptive-d-allocation/audit-result.json)
and [protocol-deviation receipt](continuous-evidence/adaptive-d-allocation/protocol-deviation.json)
are the entry points for review. The runtime driver in `source_archive` is
authoritative; the tracked driver was subsequently changed to stop after a
failed warm filter.

## What the cheap filters establish

The [warm fork](../../benchmarks/toy100/warm_equilibrium_probe.py) starts from
one scheduled, passing mode-hold state at update 1000, inheriting weights,
Adam moments, EMA and training RNG. Its 200 subsequent per-update checks test
that *state's* local quality under a new update law. They do not test cold
acquisition or prove a game equilibrium. A method that would learn a different
attractor from initialization could fail this transplant. The filter is a
conservative way to reject this configuration, not evidence that its method
family is impossible. Conversely, secant and full-J implicit responses passed
200/200 warm checks and failed cold trajectory badly; a warm pass has no
acquisition credit. The cross-only response failed four warm checks while its
unplanned cold trajectory nearly reached the terminal MSE threshold. None of
those facts justify relaxing a frozen threshold or claiming a survivor.

The 400-step [trajectory host](../../benchmarks/locked_shared/trajectory.py)
is a valid required transfer gate but is a different learning problem from
the [mode-hold host](../../benchmarks/locked_shared/mode_hold.py). It maps a
16-coordinate slow arc and learned four-coordinate particle code to a
16-coordinate fast arc, grades *identity* MSE at 0.02 and adds set coverage.
Mode hold grades generated samples against eight 2D Gaussian centers; its
12 equally weighted learned particles and narrower generated output noise
cannot exactly represent the equal-weight target mixture. Both include a
relativistic adversarial game and particle regularization, but their critic,
conditioning, gradients and success metrics differ. Trajectory failure
disqualifies a common replacement; alone it cannot show failure to acquire
the ring. A nonzero-movement floor would also be wrong: sitting at a good
target is acceptable, provided a later shift wakes learning.

The [constant-rate 96-row search](constant-lr-wave1.md) already varied base
rate, D and prior multipliers, Adam beta2 and cap strength, with no mode-hold
winner. Changing only a constant role LR ratio is therefore a poor next
explanation. The passing-state gradient diagnostic reports coherent
Adam-scaled G/prior gradients, but says nothing about whether their direction
improves sample quality. In particular, the imperfect finite-particle
representation may sustain a real critic signal at a visually acceptable
state. This matters for any criterion that equates critic separation with a
need for another generator move.

## One predeclared, different-axis experiment

[Kim et al. (August 2026)](https://arxiv.org/html/2608.10096) give
fresh-sample e-process rules for switching discriminator and generator
phases. Their 16-Gaussian experiments use fixed Adam rates and adaptive
update counts, though their original GAN game and architecture differ from
this relativistic, regularized host. We isolated their D-side
**difference-score** rule: each frozen outer update keeps the ordinary one
G/prior update and the original noise clock, while allowing 1–3 complete D
updates. After each D update, a separate seeded stream draws 16 new real,
latent, output-noise and critic-input-noise samples. For paired score gap
`d = D(real)-D(fake)`, the per-pair e-value is
`2 sigmoid(d)/(1+2 a_D-a_D²)`; a half-shrunk product forms the batch e-value,
and an e-process with betting fraction 0.5 stops D at evidence 10 or the
three-update cap. We fixed `a_D=.01`, `alpha_D=.1`, `rho_D=.5` before running.
No evaluation HQ/mode count, target center index or training age enters the
decision. The extra D update repeats the host's original D loss, gradient
penalty, fresh training batch and Adam step. G and prior Adam rates remain
`.00425` and `.0085`; D stays `.00425` in the cold arm. The warm arm uses the
scheduled prefix only to create the common state, then those same fixed
rates. Actual group-rate ranges, gradient queries and moment steps are in
each raw receipt. The independent score stream leaves every training RNG
untouched; the unit check verifies this. This is a scratch policy on mode
hold, ineligible for the production shared gate.

| Arm | Strict terminal checks | Dense checks 1001–1200 | D / G updates | D phase counts 1,2,3 |
| --- | ---: | ---: | ---: | ---: |
| Scheduled cold control | 5/5 | 200/200 | 1200 / 1200 | — |
| Constant cold control | 0/5 | 6/200 | 1200 / 1200 | — |
| Warm identity | 5/5 | 200/200 | 1200 / 1200 | — |
| Warm constant | 1/5 | 6/200 | 1200 / 1200 | — |
| Warm D allocation | 1/5 | 7/200 | 1600 / 1200 | 0,0,200 |
| Cold D allocation, unplanned diagnostic | 0/5 | 0/200 | 2509 / 1200 | 171,749,280 |

The warm identity's final full training-state hash and time-stripped metrics
match the separate scheduled cold control exactly. The unchanged controls
reproduce the known pass/failure, including warm constant's 6/200 dense
checks. The proposed arm uses 600 fresh evaluation queries and 400 extra D
gradient/Adam updates in the 200-step warm suffix. Its e-process never
crosses the threshold before the cap there, so the warm comparison is
effectively fixed 3D:1G, not proof that adaptive switching generally fails.
In the cold diagnostic, the decision genuinely varies D updates but does
not acquire sustained ring quality: all five original terminal checks fail.
Its final 6 modes/HQ 0.4973 must not be substituted for the failed terminal
window. The cold driver started automatically after the known warm failure;
this violated the requested fail-fast ordering. The run is retained as
`UNPLANNED_DIAGNOSTIC_NOT_PROMOTION`, with no gate or successor credit. No
trajectory, continued hold, translation shift, older-19 or native-3 tests
were run for this arm.

The analytic sanity check is direct: equal critic scores give negative
log-evidence, so D reaches its cap; strongly larger real than fake scores
cross evidence 10 after one batch. This verifies decision orientation, not
convergence. A key risk predicted before the run was that eight target modes
versus twelve learned particles make the small-TV null hard to satisfy even
when HQ passes. The source paper itself distinguishes its operational,
critic-induced discrepancy from population distribution equality when an
empirical law is discrete. Here both sampled laws can be continuous because
of noise, but neither a passing HQ nor this score establishes TV≤.01.
Further threshold tuning on these same failures would be another search,
not confirmation of the method.

Validation used Python 3.12.13 and PyTorch 2.13.0+cu126 on CPU. The two
focused score/RNG tests passed, and the archived runtime source hashes match
the declaration. The attempted run took about 8.5 seconds per warm child
including the shared prefix and 11.5 seconds for the later cold diagnostic;
those times are machine-local, whereas the recorded gradient and moment
counts explain the extra work portably.
