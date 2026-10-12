# Saved momentum-role analysis

Role separation is supported by the existing trainer. Saved states justify
isolating its effects; they do **not** identify which role caused the Ring16
improvement or acquisition regressions. The [compact audit](analysis.json) and
[reproducer](analyze.py) analyze six saved endpoints, construct no models, draw
no new samples and perform no optimizer updates. Python/Torch CPU RNG states
remain unchanged. Parent and momentum final training-stream cursors match for
all declared data, noise and prior streams on these three tasks.

The current [compiled memory](../../EXPERIMENT_MEMORY.md), experiment guide,
[whole five-track readout](../../r1r2-five-track-repair/README.md) and
[new density diagnosis](../../r1r2-five-track-repair/density-readout.md) retain
all original failures. The measured constant parent is
`r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9`,
revision `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775`.
Momentum is `r1r2-five-track-momentum-v1`, revision
`3c9bfbf1cb961eaef0aa32e076f906d97ff2305f066ff83dc04d3d77423f34fa`.
Both share scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
The audit checks publication-bound checkpoint hashes and records original
request/raw/sample identities without changing numerical grades.

## What the endpoints establish

The table reconstructs each last Adam parameter displacement from its saved
first/second moments, beta corrections and LR in float64. These are formula
estimates; independently captured pre/post parameter displacements are
unavailable. Parent beta1=0 first moments are its instantaneous last gradients;
momentum beta1=.5 moments are averages, so instantaneous gradients and their
history cannot be recovered from that checkpoint.

| Task | Role | Parent moment-derived displacement L2 | Momentum displacement L2 |
| --- | --- | ---: | ---: |
| Gaussian | G | 0.028710 | 0.150520 |
| Gaussian | prior | 0.056250 | 0.087570 |
| Gaussian | D | 0.094790 | 0.099530 |
| Ring16 | G | 0.123250 | 0.050820 |
| Ring16 | prior | 0.182340 | 0.071100 |
| Ring16 | D | 0.316810 | 0.152880 |

All three Ring16 roles have smaller endpoint moment-derived updates with
momentum, while Gaussian G/prior have larger ones. This rules out interpreting
these endpoints as evidence of uniformly reduced movement. Fixed saved-output
critic gradient norms also differ: Ring16 means 0.258244→0.097265, Gaussian
0.008400→0.102970. Those are gradients of deterministic saved critics on their
own saved output panels; they are neither matched training batches nor raw
training parameter gradients.

The parent acquired Gaussian at an earlier observation with CDF KS 0.024177
(update 417). Momentum's best KS is 0.053078 at update 1,000, missing <=0.05;
its endpoint mean error and std ratio already meet their respective bounds.
A small endpoint mismatch and a lost independently confirmed acquisition gate
are different claims.

Both word endpoints are collapsed with effectively stationary G. Parent G
first-moment norm is 3.13e-19 and reconstructed displacement 4.08e-16; momentum
values are 7.18e-13 and 1.16e-8. E/prior/D remain active. On the explicit fixed
five-row panel, momentum maps every saved prior mean and every canonical
reconstruction to `grape_`; parent maps rows to apple/grape and one malformed
word. Momentum's minimum prior-row separation is 0.060699 versus 0.318505;
minimum encoded canonical separation is 0.014612 versus 0.312149. Both fail
word acquisition. The saturated G endpoint suggests a difficult late escape,
but does not establish whether D, G/E or prior momentum caused early collapse.
These deterministic panels do not replace the original sampled word gates.

For Gaussian and Ring16, 58.98% of parent prior first-moment coordinates are
exactly zero; momentum has none. Moment history therefore extends nonzero
update directions beyond an instantaneous-gradient-only rule. Actual sampled
row IDs and momentum's last raw gradient support were not retained, so this
is not a measured count of unsampled rows being moved. The finite five-row
word prior does not have the same sparse-row argument.

Momentum's density gain is real but incomplete: best Ring16 covariance error
1.098351, endpoint 1.185580, both above 0.85; minimum eigen ratio 0.052811 also
misses >=0.15. Reducing tangent stretch while retaining radial spread remains
necessary. Neither role separation nor an AMSGrad interaction has a proven
complete repair here.

## Supported recipe ownership

Native [recipe factories](../../../../particlegan/recipes.py) consume
`Recipe.betas` for G/E, `Recipe.d_betas or Recipe.betas` for D, and explicit
`Recipe.prior_betas` for sampled learned prior groups. The
[word constructor](../../../../benchmarks/toy_audit/api_images.py) and
[behavior binder](../../../../experiments/forge/behavior_adapters.py) consume
the same roles. There is no separate E beta field; E follows G.

Two-pole direct generated coordinates consume the **G beta factor**, although
rate bookkeeping labels their group `forge_role=prior`. Its constructor sets
no prior-betas override, and the native Adam branch returns before the
formulation-only direct-particle response. The behavioral receipt explicitly
marks `prior_betas_consumed=false` for direct coordinates. Sampled prior betas
are inapplicable to two-pole and unsampled token cohorts. With `beta2_end=null`,
the [schedule](../../../../particlegan/recipe_schedules.py) does not overwrite
any role's constant betas. Native moment counters remain checkpoint-owned.
No trainer or task change is required for D/G+E/prior beta1 factors.

## Two explicit hypotheses

From the exact measured all-.5 momentum recipe, retain beta2=.99, paired
R1/R2 coefficient .1 every update, LR .0085, learned prior and every other
recipe/task/protocol field:

1. Set only `d_betas=[0,.99]`, retaining G/E/prior beta1=.5. This tests whether
   critic momentum contributes to the Gaussian acquisition miss and early
   word collapse while generator smoothing retains the Ring16 gain. Falsify a
   complete repair if any original ordinary or clock gate fails; specifically
   require independently confirmed Gaussian/word acquisition and full Ring16
   covariance <=.85, eigen ratio >=.15 and five terminal passes.
2. Set only `prior_betas=[0,.99]`, retaining D/G/E beta1=.5. This tests whether
   prior history and code motion contribute to acquisition regressions while
   network momentum retains useful geometry. The same unchanged complete-suite
   requirements falsify the repair. Prior pair separation or a smaller final
   covariance alone supplies no qualification.

Confidence in either single-role intervention is limited by the coupled game
and the parent's own word failure. A preregistered finite D/G+E/prior factorial
with exact all-zero/all-.5 reuse distinguishes those hypotheses and their
composition more clearly. The G-only-momentum subset is an explicit new global
recipe, not a per-task mixture. All-.5 plus AMSGrad is a supported interaction,
but separate AMSGrad word acquisition and momentum density improvements cannot
be combined as existing qualification. No admission blocker was found in the
examined beta ownership paths. Root owns declarations, budgets and execution;
this analysis authorizes or launches no training.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/momentum/analyze.py > reports/forge/r1r2-role-repair/momentum/analysis.json
```
