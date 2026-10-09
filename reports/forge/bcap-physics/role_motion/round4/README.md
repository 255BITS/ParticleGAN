# Round4: generator/prior finite motion attribution

**Role balancing improves native precision and Gaussian retention, but worsens
unequal-width full covariance. No failing task becomes a full sustained PASS.**
The [completed PR371](https://github.com/255BITS/ParticleGAN/pull/371) retains
the original winner and its archived 7/21 qualification. Stop this exact revision
as a global repair; keep the measured role attribution and scoped gains.

| Unchanged task | Matched winner | Role-balance candidate | Retained result |
| --- | --- | --- | --- |
| Two-pole | PASS | BLOCKED | Candidate frozen component host lacks the controller; zero candidate spend |
| Gaussian smoke | PASS, first confirmed 375 | PASS, first confirmed 792 | Acquisition slows; full 1,000 updates retained |
| Gaussian stability | FAIL | FAIL | Stationary 2/72→62/72; shifted hold 0/24→13/24; deadline reacquisition fails both; final KS .320623→.045352 |
| Native grid100 | FAIL, holdout precision .240720 | FAIL, holdout precision .637260 | Forecast≥.48 observed; all five terminal accuracy checks fail; mass TV .147180→.187360 |
| Unequal width | FAIL, covariance 6.287559 | FAIL, covariance 11.472665 | Zero passing checks both; retained original full covariance≤.85 bound |
| Two broad | PASS, covariance .385581 | PASS, covariance .123979 | Both 22/24 passing checks and suffix 22, first pass 150; passing behavior retained |

Both arms are 2 PASS / 3 FAIL on their five mutually executable tasks. Control's
additional two-pole PASS and candidate's BLOCKED cell stay visible; they do not
change that comparison. [Final metrics and full temporal grading](results.json),
[certified source/protocol receipts](provenance.json),
[controller and density diagnostics](controller-and-density.json), and
[scorer controls](scorer-controls.json) bind these conclusions.

The initial exact-source probes identify generator motion as dominant in the
examined endpoint next-update cohorts. Native network/prior/joint RMS travel
is .245671 / .036805 / .268698, with nonlinear cross RMS .002467. Network center travel
is .245057 while jitter deformation is .005016: most proposed native travel is
center movement. In transport-v2's narrow spill diagnostic, network outward
motion dominates on the three sampled spill rows. That very small cohort is
explicitly limited; it does not establish population attribution or explain
every prior trajectory. All five disposable copies restore optimizer and streams
exactly and preserve their original files.

During the new candidate's 7,000 native updates, every update is limited with zero
rejections and maximum measured network/prior RMS ratio .999996. Average proposed
G RMS .182935 becomes .016700, while the retained prior RMS is .022968; actual joint
RMS is .030970. Average G/prior jitter deformation is .000279 / .000811, showing the
remaining deformation is now larger in the prior role. This finite bound is
verified; the holdout density still fails. Native local Jacobian total variance
median increases 2.976159→3.315985 times target variance even as precision improves.
Uncensored saved-center covariance error also worsens 8.597249→45.407746. These
are separate diagnostics: the official accuracy moments censor by quality radius
and cannot substitute for full-population width or tail quality.

Unequal-width four-sigma core covariance improves .444178→.296830 while full
covariance worsens 6.287559→11.472665. First two full errors are 17.901299 / 20.179314,
with spill .055459 / .082707; mass TV remains .294189. Local Jacobian variance median
is only .007918 times assigned target total variance. In the saved-center census,
the two narrow components have center covariance trace 15.300945 / 21.335876 times
target trace, versus linearized kernel contributions .062597 / .020831. Center
spread dominates this fixed-assignment diagnostic. The kernel approximation
omits nonlinear jitter and changing assignments; actual served scores remain
authoritative. Low local Jacobian spread does not imply correct served density.

The new evidence supports distinguishing center transport and within-kernel
deformation before another mechanism. It rejects this exact RMS balance as a
global repair: slowing G can improve retention/precision while leaving allocation
and center spill uncontrolled. Retain the winner/default, preserve transport-v2's
separate rare-density cohort, and inspect saved center-population dynamics before
any separately preregistered successor. No sweep, seed repeat, additional arm,
unchanged continuation or adoption follows this completed comparison.

## Certification, cost and actual training

Executed commit `313b437cd364a38c618e5d959c7791488c2a1bf7`; both arms share source
digest `6c28f92aab96366aa6fd3b7d0bae3929ea83ebd458c719365259d5da70084ebb`.
All 1,201 executed source files match the published scientific implementation.
Initialization/model/prior, task declarations, recipes, actual seen batches and
consumed training RNG streams match in all five runnable paired tasks. Evaluation
streams remain separate; continuations restore their own certified 1,000-update
smoke prefix and complete the original 6,000-step horizon. Gaussian smoke may
PASS from an earlier confirmed state even if its final endpoint misses KS;
control's final smoke KS .071842 does not rewrite its valid acquisition grade.

All 11 runnable jobs complete once, producing 11 actual-training GIFs. Worker cost
is 1,495.309652 seconds (24.92 worker minutes), with zero scientific retries,
zero INVALID/INCOMPLETE attempts and no reservation or live worker remaining.
Planned main allowances total 16,440; executed allowances 16,140 exclude the 300-second
unsupported candidate cell. The saved-probe/software ceilings bring planned
full allowances to 17,280, below the 18,000-second track limit. Completed disposable probes
add five diagnostic updates; their measured method runtimes are separately in
the probe receipt, with process startup overhead outside that measurement.
The single sampler refusal occurs before any update. Contention and added
forward probes are accounted work, not an optimizer-speed comparison.

[84 meaningful software checks](software-verification.json) pass; declarations
validate. Four independent oracle controls PASS and all four center-collapse
controls FAIL. [Media receipts](media/index.json) verify fixed-index saved arrays
and render without training, model calls or new generated draws. Final readouts
close both subscriptions; the unsupported cell leaves the candidate study
decision incomplete rather than being removed. Original task/qualification/
telemetry hashes remain byte-identical in [preservation.json](preservation.json).

| Actual task | Matched control GIF | Candidate GIF |
| --- | --- | --- |
| Two-pole | [Measurements](media/control/two_pole.gif) | Explicitly BLOCKED |
| Gaussian smoke | [Target and draws](media/control/gaussian1d_smoke.gif) | [Target and draws](media/candidate/gaussian1d_smoke.gif) |
| Gaussian stability | [Hold and shift](media/control/gaussian1d_stability.gif) | [Hold and shift](media/candidate/gaussian1d_stability.gif) |
| Native grid100 | [Density and gates](media/control/grid100.gif) | [Density and gates](media/candidate/grid100.gif) |
| Unequal width | [Density and gates](media/control/vector_unequal_width.gif) | [Density and gates](media/candidate/vector_unequal_width.gif) |
| Two broad | [Density and gates](media/control/vector_two_broad.gif) | [Density and gates](media/candidate/vector_two_broad.gif) |

Reproduce compact metrics, media and diagnostics from exact retained artifacts:

```sh
QUEUE=/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/role_motion/queue
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/role_motion/round4/publish.py --queue-root "$QUEUE" --media
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/role_motion/round4/analyze.py
```

The frozen declaration and rationale below preserve the pre-run contract.

One global role-aware control is preregistered against the exact winning BCAP
configuration. This is a bounded diagnostic comparison, with unchanged full
gates, and supplies no ordinary Tier2 qualification or default promotion.

The [five saved-source probes](saved-role-probes.json) restore original winner
and transport-v2 endpoint checkpoints using their exact archived sources. One
next-batch update is previewed on each disposable copy, then the complete model,
optimizer and consumed-stream state is restored exactly. The native network
output RMS is .245671, prior RMS .036805, joint RMS .268698. Across all five
probes, network RMS is 4.4–6.8 times prior RMS and network jitter deformation is
larger. These are finite next-update measurements, not the original preceding
training step, full density grades or causal explanations of every failure.
Posthoc target component/core/spill labels only describe diagnostics.

## Mechanism and falsifier

Let z=c+epsilon be the already consumed generator-phase latent batch, and let
(theta1,c1) be the actual ordinary momentum0 DualNorm proposal from(theta0,c0).
Measure p=RMS[G_theta0(c1+epsilon)-G_theta0(c0+epsilon)] and
g(s)=RMS[G_(theta0+s*(theta1-theta0))(c0+epsilon)-G_theta0(c0+epsilon)].
Retain the whole c1 update. Use s=1 if g(1)<=p; otherwise start at p/g(1)
and halve up to 24 times until the actual finite g(s)<=p. A zero response or
exhausted search restores G exactly. Unchanged s=1 never recomposes floating
parameters. The public `GANTrainer(role_motion_balance=True)` owns this path.
The prior remains learned with fixed task width/uniform weights, and receives
its full ordinary optimizer proposal. All original losses/rates are unchanged.

The constraint balances two degrees of freedom using the observed composition
G(c), without labels, target sigma, evaluator centers, extra batches or draws.
It measures center travel, within-kernel finite deformation and nonlinear cross
interaction separately. It bounds only network motion on consumed rows; the
joint nonlinear response and served density remain unconstrained. A small prior
gradient can suppress useful G progress. The cap can trade overshoot for stalled
learning or contraction. No density, curvature or convergence guarantee follows.

[Bolte, Sabach and Teboulle's PALM](https://bolte.perso.math.cnrs.fr/BST2013.pdf)
treats coupled variable blocks with block-specific proximal steps under explicit
minimization assumptions. It motivates distinguishing roles, but this normalized
alternating GAN rule is not PALM and does not inherit that theorem. The rule's
finite measured inequality is an implementation property; task fidelity is an
experimental question.

Two ready schema-v3 studies freeze one candidate and the exact winner as its
matched control on the same source/runtime: two-pole, Gaussian smoke and each
arm's own eligible stability continuation, nativegrid100, unequal-width and the
passing broad-vector guardrail. Candidate two-pole is explicitly unsupported
because its frozen public-component host lacks this controller. Both original
task/gate definitions remain unchanged. Smoke failure blocks its own dependent
continuation; other diagnostic peers still complete. Seed0, public deterministic
initializer, architecture/data/prior/sampling/update limits and scoring cadence
remain matched. The frozen vector/Gaussian adapters actually reuse one real
tensor for D/G; that law is preserved and disclosed.

Preregistered native forecast: precision>=.48; precision<.30 falsifies it.
Unequal-width complete full-covariance/suffix gates and broad PASS are additional
required review outcomes; endpoint bounds do not replace sustained grading.
Exactly one comparison is authorized; no seed runs, rate search, continuation,
automatic second candidate or unchanged merge rerun follows this study.

## Budget and execution

The full main ceiling is 16,440 seconds (8,220 per arm). Five completed saved probes
plus one pretraining sampler refusal reserve 720 seconds; meaningful tiny public
API software checks reserve 120 seconds, leaving total 17,280 below the 18,000-second track
ceiling. These diagnostic updates never fill scientific grades. The initial
vector probe supplied a CUDA generator to the CPU sampler; it refused before
any update. The corrected probe binds the original checkpointed CPU stream.

The independent public Queue/drain runner uses one freely shared GPU worker,
with the full-compilation completion callback disabled. All arms are pushed
before enqueue. Bulk logs, JSONL streams and checkpoints remain outside Git.
Tail execution with:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/role_motion/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/role_motion/queue/role-motion-round4-v1/progress.jsonl
```

Use `/home/martyn/dev/ParticleGAN/.venv/bin/python` with `PYTHONPATH=.` from this
worktree. `run.py plan|enqueue|drain --queue-root QUEUE` reproduces the declared
workflow. The ready declarations are under `configs/forge/studies/role-motion-*`.
The control-side unsupported-cell study admission follows the earlier
transport-v2 fix, retaining known API refusals and every original task identity.
Actual-training GIFs and compact certified final metrics are published above;
saved probes remain contextual, not a third matched arm.
