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

---

## Results (appended after the runs)

**Verdict: REFUTED at stage 1, under rule 1.** With rho = 0.03, the relative tether (R) failed `mode_hold` and also
failed `ring16_acquisition`, which the control passes. As preregistered, stage 2 (word smoke and hold) and
stage 3 were not run. Both context checks reproduced #390 on this source: C FAILS `mode_hold` and F PASSES it.

Frozen source digest `120d9cb44c53bfa5981ff163395e322cdb25707f0c398cd891edff94e17583ed` (commit `1f89a824`),
used by all three arms. Seed 0, CUDA RTX A6000 (GPU 0, shared), one worker.

| Arm | Test | Gate | Terminal passing suffix (need) | Passing observations | Endpoint metrics (not the verdict) | Wall s |
| --- | --- | --- | --- | --- | --- | ---: |
| C control | mode_hold | FAIL | 3 (>=5) | 10/24 | modes 8>=8, hq 0.989>=0.9 | 41.6 |
| F fixed .01 | mode_hold | **PASS** | 7 (>=5) | 10/24 | modes 8, hq 0.999 | 66.9 |
| R cap 0.03 | mode_hold | FAIL | 0 (>=5) | **0/24** | modes 6 (<8), hq 0.863 (<0.9) | 41.2 |
| C control | ring16_acquisition | PASS | 26 | 26/96 | cov err 0.497<=0.85, min eig ratio 0.242>=0.15 | 33.6 |
| F fixed .01 | ring16_acquisition | PASS | 9 | 10/96 | cov err 0.726<=0.85, min eig ratio 0.325 | 46.4 |
| R cap 0.03 | ring16_acquisition | FAIL | 0 | **0/96** | cov err **1.343 > 0.85**; modes 16, hq 0.917 pass | 32.9 |

Definitions: "terminal passing suffix" counts consecutive passing scheduled checks at the end of training.
A sustained gate PASS needs at least the task's minimum, e.g. 5 for `mode_hold`. "Passing observations" counts
checks where every metric passed, anywhere in the run. "cov err" is ring16's per-component covariance error, i.e.
how far each generated ring component's shape is from the true component shape.

`mode_hold` curves (step, modes found of 8, high-quality fraction), every 100 steps:

- C: (650, 8, .75) (750, 8, .91) (850, 8, .93) (950, 8, .94) (1050, 8, **.87**) (1150, 8, .93)
- F: (650, 8, .87) (750, 8, .94) (850, 8, .89) (950, 8, .98) (1050, 8, 1.0) (1150, 8, .99)
- R: (250-1050, **7**, .57 to .98) (1150, 6, .89). R reaches high quality but never covers the 8th mode.

GPU seconds: 262.6 s of training wall time across six attempts (about 0.07 GPU-hours). One coordinator
process also spent about 20 CPU-minutes in the post-completion memory compile before I stopped it. No training was
lost.

## Interpretation

- The capped tether is not an interpolation between C and F. It fails both gates outright, with zero passing
  checks, while C and F each pass ring16 and record 10 passing `mode_hold` checks. So these gates react
  non-monotonically to small changes in prior force. #390 saw the same in ring16: the .001 and .1 fixed tethers
  failed it while .01 passed. One seed cannot separate "the cap removes the useful part of the force" from
  "any perturbation of this size moves BCAP to a different trajectory".
- The mechanism question from #393 (does word hold recover?) remains **untested**. The rule stopped the run before
  the word tests because the cap already lost two gates the fixed tether keeps.
- Practical conclusion: with rho = 0.03, the relative cap is not a way to keep the tether's mode_hold repair. Keep
  the feature opt-in (default 0, byte-identical to the fixed tether). Do not adopt it.

## Exact commands

```sh
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python
LEASE=/tmp/claude-1000/-ml2-hypergan/baedb70c-2c24-4e91-92d1-2f2bed2d913c/scratchpad/dig/gpu-lease.sh
Q=/ml2/hypergan/dig-20261015/relative-tether/queue
for a in control:bcap-prior-covariance-control-v1 fixed:dig-relative-tether-fixed-v1 cap:dig-relative-tether-cap-v1; do
  OMP_NUM_THREADS=1 $PY -m experiments.forge --queue-root $Q enqueue ${a#*:} --study dig-relative-tether-${a%%:*}-stage1-v1
done
$LEASE bash -c 'FORGE_SKIP_COMPLETION_COMPILE=1 OMP_NUM_THREADS=1 '$PY' -m experiments.forge --queue-root '$Q' drain --gpus $LEASED_GPU --workers-per-gpu 1 --campaign dig-relative-tether-v1'
tail -F $Q/events.jsonl          # live per-check metrics
```

Request ids (stage 1): C `b39caa63006169537247aaec`, F `d81a8ff8b0c9a26c94c3c31d`, R `63a54704f290ef385d3fb34e`.
Attempts: C `3e0b4949…` / `8248410c…`, F `9f3074ac…` / `2b5249c3…`, R `84e06270…` / `ac521035…` under
`$Q/dig-relative-tether-v1/`. Stage 2/3 requests were frozen on the same source digest in `queue-stage2` and
`queue-stage3` but never drained, because the rule stopped at stage 1.

## Harness issues found, and what I did

1. `experiments/forge/knowledge.py` used `deepcopy` without importing it, so `NameError` crashed drain's
   post-completion compile. Fixed by adding the import.
2. In the same compile cache, `cached_attempts` called `_attempts(r, t)`, but `_attempts` takes one argument
   (`TypeError`). Fixed.
3. Once fixed, that post-completion compile rebuilds every goal leaderboard. It ran about 20 CPU-minutes after a
   41-second job, blocked the next launch and rewrote tracked leaderboards in the checkout. I added an opt-in
   `FORGE_SKIP_COMPLETION_COMPILE=1` (coordinator only; it changes no training code and no default) and reverted
   the rewritten leaderboards. The coordinator fixes were made after all requests were frozen, so they do not
   change the frozen source of any attempt.

## Caveats

- One seed (0), one rho (0.03), no tuning. A different rho, or a cap on the *displacement* rather than the
  gradient, was not tried.
- The cap acts per backward pass on the full gradient of each latent table. `g_other` includes every non-tether
  term in that backward, not only the adversarial loss. Under DualNorm, the optimizer normalises the combined
  update afterwards. The in-loop scale `s` is kept in `MomentTetherRegularizer.relative_cap_stats`, but no
  receipt exports it, so the actual in-training relative force was not measured.
- The pre-existing failure `tests/test_forge_dualnorm_smoothing_search.py::test_other_families_do_not_claim_an_active_smoothing_axis`
  fails in this checkout with `optimizer_smoothing requires optimizer_family='dualnorm'` (it builds the `bcap_adam`
  preset with other optimizer families). It goes through none of the code this branch touches. I did not re-run it
  on a clean develop checkout.
