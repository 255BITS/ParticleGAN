# Relative-force tether: can a capped prior tether keep mode_hold without breaking word hold?

Status: PREREGISTERED (rule written 2026-10-15 before any training). Results are appended below the rule.

## Background and hypothesis

#390 added an opt-in "moment tether" (`prior_regularizer=moment_tether`): a penalty that pulls the learned
latent table's mean and covariance back toward their initial values. At `prior_reg=0.01` on the BCAP leader it
turned `mode_hold` FAIL -> PASS but turned `five_word_joint_hold` PASS -> FAIL. #393 measured, on saved states,
the tether's gradient on the latent table relative to the adversarial gradient on the same table ("relative
force"): 0.011 at the mode endpoint, 0.075 at word acquisition, 0.341 at the word endpoint. The word task's
adversarial signal shrinks as it converges, so a fixed tether comes to dominate it.

Hypothesis: capping the tether's force at a fixed fraction rho of the adversarial force keeps the mode repair
(relative force there is ~0.011, below the cap) and removes the word regression (where the fixed tether is
2.5x to 11x above the cap), without losing ring16.

## Mechanism (opt-in, portable)

New recipe field `prior_tether_relative_cap` (default 0 = exactly the old fixed tether, no hooks, no extra
autograd). When > 0 and `prior_regularizer=moment_tether`, for each backward pass and each latent table z:
`g_z = g_other + s * g_tether`, `s = min(1, rho * ||g_other|| / ||g_tether||)`, where `g_tether` is the tether's
own gradient on z (including the outer `prior_reg` coefficient) and `g_other` is everything else in that backward
(normally the adversarial generator objective). Only gradient norms on the latent table are used: no samples,
RNG, task statistic or optimizer state. Declared value: **rho = 0.03** (between the mode regime 0.011 and the
word-acquisition regime 0.075; chosen once from #393, not tuned).

## Arms (one frozen source, seed 0, CUDA RTX A6000, own queue root)

| Arm | Candidate | Change vs C |
| --- | --- | --- |
| C (control) | `bcap-prior-covariance-control-v1` | none (the #390 matched BCAP direction-blend leader control) |
| F (fixed tether, #390 reproduction) | `dig-relative-tether-fixed-v1` | `prior_regularizer=moment_tether`, `prior_reg=0.01` |
| R (relative tether) | `dig-relative-tether-cap-v1` | F + `prior_tether_relative_cap=0.03` |

## Preregistered decision rule

Verdicts come from each task's Forge sustained gate (terminal-suffix passes), never endpoint metrics.

- Stage 1: `mode_hold`, `ring16_acquisition` on C, F, R.
- Stage 2: `five_word_joint_smoke` + dependent `five_word_joint_hold` on C, F, R.
- Stage 3 (only if SUPPORTED so far and budget remains): `gaussian1d_smoke`, `two_pole`, `unused_token_hold`,
  `ae_gan_hold` on C and R.

1. If R FAILS `mode_hold`, or R FAILS `ring16_acquisition` while C passes it: **REFUTED** (the cap loses the
   repair or loses ring16). Stop after stage 1.
2. Otherwise run stage 2. **SUPPORTED** if R PASSES `five_word_joint_hold` (and `five_word_joint_smoke`).
   **REFUTED** if R FAILS `five_word_joint_hold`.
3. Context checks (reported, they do not flip the verdict on R): C should FAIL `mode_hold`; F should PASS
   `mode_hold` and FAIL `five_word_joint_hold`. If C passes `mode_hold`, the repair is moot on this source and the
   verdict becomes INCONCLUSIVE. If F passes word hold, the #390 regression does not reproduce here; an R pass is
   then reported as "supported but not discriminating".
4. Stage 3 losses (R fails a test C passes) are reported as costs; they downgrade SUPPORTED to "supported with
   cost" in the conclusion.
5. INCOMPLETE/TIMEOUT/BLOCKED cells that the rule needs make the verdict INCONCLUSIVE for that cell; no reruns
   with other seeds. A wall-clock TIMEOUT of the 900 s word smoke on a shared GPU is reported as infrastructure,
   not as a recipe FAIL.

Budget: at most about 2 GPU-hours of training. One job per GPU through the workflow lease helper.
