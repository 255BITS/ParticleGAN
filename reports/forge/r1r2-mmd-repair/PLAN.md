# One fixed output-MMD contrast — preparation only

**Status: draft, paid budget pending.** Prepare one opt-in global structural
objective; no paid run, qualification or default adoption follows from this plan.
The [saved-state research](../r1r2-next-repair-research/mmd/README.md) supplies a
modest distinguishing prerequisite: fixed-scale MMD contracts mean radial width
on overwide learned-energy outputs where energy still expands it. It does not
predict the parameter-space training trajectory. The
[least-squares review](../r1r2-next-repair-research/least-squares/README.md)
does not justify a competing arm.

Use the exact native non-saturating AMSGrad m110 parent. The sole exercised
global delta is **`output_mmd_weight: 0 → 1`**. Keep the fixed Gaussian kernel
standard deviations `(.0625, .125, .25, .5, 1, 2, 4)` in original task output
units, with equal weights. Preserve native non-saturating loss, energy weight 0,
positive paired real/fake R1/R2 coefficient .1 every update, LR .0085,
network betas (.5,.99), prior betas (0,.99), native Adam/AMSGrad, learned prior,
prior regularizer 0 and all other actual parent fields. There is no coefficient
or bandwidth grid and no claim of an optimized or unit-invariant strength.

For existing G-phase fake outputs X and detached existing real outputs Y, add

`mean(i != j) k(x_i,x_j) + mean(i != j) k(y_i,y_j) - 2 mean(i,j) k(x_i,y_j)`.

The result is a signed squared-MMD U-form: retain both fake derivatives and
duplicate observations; remove only within-panel diagonal indices. Apply no
square root, clamp, normalization, jitter, feature encoder, median bandwidth,
target component label, altered coupling or additional draw. Priors remain
learned and receive auxiliary gradients through the original G path. Direct
pole coordinates and token student embeddings retain their original generator
ownership; their original task declares that no prior is sampled. Words use
the original probability output coordinates, particle-cloud exception and
native joint inverse branch. Marginal matching cannot replace reconstruction,
unused-token hold or any other original host objective.

Preserve all original task and protocol contracts: six ordinary tasks
(Gaussian, pole, token, AE, Ring16 and words), plus this recipe's own clock.
Hold architecture, target/data law, ordered batches, protocol seed 0, public
deterministic initialization, initial prior distribution/width/capacity,
sampling, full update budget, evaluation cadence and sustained gates fixed.
Original Ring bounds remain covariance error ≤ .85, minimum eigen ratio ≥ .15,
HQ ≥ .85, 16 modes, mass TV ≤ .15 and at least 4096 evaluation samples, with
the original five-observation passing terminal suffix. Full word training
remains 20001 updates. A failure on any required ordinary task or own clock
rejects the whole repair; each independent runnable Tier 1 peer must finish.
No task-specific recipe selection or extra tier is authorized.

The public addition must be default-zero compatible, use caller-owned panels,
consume no randomness and preserve isolated constructor/data/training/evaluation
streams. It needs finite mathematical/autograd checks, strict signed accounting
and recipe-bound checkpoint validation, unchanged default metadata and original
evaluator pins. Explicit external bindings inherit the original component and
word steps and reuse their existing output hook. Native trainer and own-clock
paths must consume the same recipe. Restore must validate separate MMD state
before mutating original parent state. Unsupported or combined component
objectives must block explicitly. Software fixtures are separate cohorts and
confer no task-quality or clock qualification.

Use exactly two roles, fresh `non_saturating` control and fresh `mmd_u`
treatment, seven original cases each. This public addition changes scientific
source; archived NS, energy and frozen-prior certificates retain their original
source and cannot fill the fresh control. Bind source/runtime, resolved global
and all seven host recipes, exact task conditions, named RNG streams and source
commit before execution. Require independent readiness/registration review and
commit the final execution inputs before any job admission.

The shared queue's [read-only budget audit](../r1r2-next-repair-research/budget-audit.json)
records 238 paid attempts / 7168.248061007995 seconds, zero active reservations
and the existing **8000-second** aggregate cap. The remaining
**831.7519389920053 seconds** cannot reserve one unchanged **900-second** word
task. Full allowances stay **2520 seconds per role / 5040 total**:

| Task | Full allowance, seconds |
| --- | ---: |
| Gaussian | 120 |
| Pole | 300 |
| Token | 300 |
| AE | 300 |
| Ring16 | 300 |
| Words | 900 |
| Own clock | 300 |

One complete fresh-source comparison therefore needs aggregate cap
**12209 seconds**, an increase of **4209 seconds**. The
[draft manifest](manifest.json) and both draft studies explicitly retain pending
authorization. Do not raise the cap, reset accounting through another queue,
shorten the word task, invent cache equivalence, launch prefixes or retry
unchanged runs. A later explicit budget decision must be recorded separately;
this software preparation leaves the shared queue and historical guards intact.

After authorization and final registration, one worker per A6000, one CPU thread,
deterministic execution and TF32 off remain the comparison resource contract.
Apply the shared 8 GiB archive ceiling and 5 GiB free-space floor. Store bulk
stdout, JSONL traces, checkpoints and tensor dumps outside Git. Commit compact
metrics, original receipts, reproduction sources and actual-training GIFs.
Keep new logs easy to tail under
`/mnt/ml7tb/ParticleGAN-forge/r1r2-mmd-preparation-20261012/`.
