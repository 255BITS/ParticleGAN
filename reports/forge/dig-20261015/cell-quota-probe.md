# Dig 2026-10-15, lead 4: cell-quota-min-change probe (`cell-quota-probe`)

**Verdict: REFUTED.** `cell-quota-minimum-change-forward-kl-g-v1` decisively fails `ring16_acquisition`
(0 of 96 checks pass) while the matched BCAP DualNorm control passes it (26 consecutive passing checks).
The recipe also cannot run two_pole or five_word_joint_smoke at all. On one frozen source, the
ring16 vs held-test conflict from the review shows up as a clean matched pair:

| recipe | ring16_acquisition | mode_hold | gaussian1d_stability |
|---|---|---|---|
| cell-quota candidate | **FAIL** (0/96) | **PASS** (21/24, terminal streak 21) | **PASS** (72/72 stationary, 24/24 shifted hold) |
| BCAP DualNorm control | **PASS** (26/96, terminal streak 26) | **FAIL** (10/24, terminal streak 3 of 5 needed) | **FAIL** (2/72 stationary, 0/24 shifted hold) |

## Terms

- **ring16_acquisition**: 2-D ring of 16 Gaussian clusters, 1,600 updates, 96 checks. A check passes when all of
  these hold: 16 modes, mass_tv <= 0.15, hq >= 0.85 (fraction of samples within 3 sigma of a cluster),
  component_covariance_error <= 0.85, component_min_eigen_ratio >= 0.15. The test passes when the last 5 or more
  checks in a row pass (the "terminal streak", also called the passing suffix).
- **Sustained gate**: Forge grades the run's whole check curve, not the final metrics. Every verdict below is
  Forge's graded `gate_status`.
- **Cell objective (cell-quota recipe)**: an extra generator loss. It splits the observed training data into
  "cells" (one per latent MoG component), estimates each cell's mean and covariance with an exponential moving
  memory (`exponential8`), assigns generator components to cells under integer quotas, and pulls the generated
  per-cell mean and covariance toward the data's (forward KL). **minimum_change** quota rounding keeps the previous
  integer allocation where possible, instead of re-rounding by largest remainder each step.
- **BCAP DualNorm control** (`bcap-default-baseline-control-v1`): the BCAP recipe with the DualNorm optimizer
  (each update rescaled to a fixed size, per-offset), constant rates. It is also the root parent recipe of
  the cell lineage.
- **NSGDA attribution arm** (`cell-quota-probe-nsgda-parent-v1`): the candidate's recipe with every cell field and
  the gradient-signal attenuation removed. That leaves the control's recipe with the optimizer swapped to global NSGDA.

## Hypothesis and preregistered rule

Preregistered in `/ml2/hypergan/dig-20261015/cell-quota-probe/PREREGISTRATION.md` before any run (addenda A and B
were written before their runs). SUPPORTED if the candidate passes ring16 AND mode_hold AND gaussian1d_stability.
REFUTED if the control passes ring16 and the candidate fails it (stop; record failure mode). INCONCLUSIVE if the control
also fails ring16, or on ERROR/TIMEOUT.

## Porting (does it run on develop?)

No. develop has none of the code. The recipe was built in the Astra research workspace
(`/ml2/codex-storage/future-workspaces/2026-10-12/task-81126997-1552-4830-8720-0d10c0853d6f/astra-remaining-mechanisms/`,
`source` tree) and never merged. Minimal faithful port on this branch:

- copied as-is: `particlegan/cell_gaussian/` (9 files) and `particlegan/optim/gradient_signal.py`;
- additive hunks only from the Astra diff in `particlegan/recipes.py` (11 default-off Recipe fields + validation),
  `particlegan/training.py` (cell objective hook + checkpoint state), `particlegan/optim/dualnorm.py` (gradient-signal
  option), `experiments/forge/api.py` (extension spec + "requires the public GANTrainer path" blocker) and
  `experiments/forge/boundaries.py`. Astra's removal of BISPF was **not** ported. develop's BISPF code is untouched;
- `particlegan/recipe_compat.py`: the new default-off fields are projected out of recipe identities, so every existing
  configuration and receipt identity is unchanged (`forge validate` passes; the dualnorm test files pass, 114 tests).

Fidelity check: on the ported source the candidate reproduces its original Astra results. gaussian1d_smoke first
confirms at step 292 with 18/24 passing checks, identical to the Astra record
(`reports/forge/test-performance-evidence/astra-cumulative-24.json`, cell 1). mode_hold and gaussian1d_stability PASS as
before.

## Commands

Every Forge call ran via a wrapper equal to
`OMP_NUM_THREADS=1 <python312> -m experiments.forge --root <worktree> --queue-root /ml2/hypergan/dig-20261015/cell-quota-probe/queue ...`

```
forge validate
forge plan cell-quota-minimum-change-forward-kl-g-v1 --view cell-quota-probe-ring16-diagnostic-v1 --through-tier 1 --device cuda
forge plan cell-quota-minimum-change-forward-kl-g-v1 --view cell-quota-probe-rest-diagnostic-v1  --through-tier 1 --device cuda
forge enqueue cell-quota-minimum-change-forward-kl-g-v1 --study cell-quota-probe-ring16-candidate-study        --device cuda  # 74d3b15f
forge enqueue bcap-default-baseline-control-v1          --study cell-quota-probe-ring16-control-study          --device cuda  # 47a24901
forge enqueue cell-quota-probe-nsgda-parent-v1           --study cell-quota-probe-ring16-attribution-study      --device cuda  # 67ec7fd0
forge enqueue cell-quota-dualnorm-probe-v1               --study cell-quota-probe-ring16-dualnorm-variant-study --device cuda  # a3ef6a7f
forge enqueue cell-quota-minimum-change-forward-kl-g-v1 --study cell-quota-probe-held-candidate-study          --device cuda  # e9c296ac
forge enqueue bcap-default-baseline-control-v1          --study cell-quota-probe-held-control-study            --device cuda  # e4ca5b7e
gpu-lease.sh drain1|drainloop <request>   # = forge drain --gpus $LEASED_GPU --workers-per-gpu 1 --request <request>
```

Seed 0 (protocol-fixed) for every run. CUDA deterministic, RTX A6000. Source digest `eb24211d...` for every run except
the attribution arm (`c5d9fc58...`), whose snapshot differs only by a one-line import in the reporting module
`experiments/forge/knowledge.py` (since reverted). Training code was identical.

## Per-test results

| stage | recipe | task | verdict | passing checks / terminal streak | final metrics (failing gates in bold) | GPU s |
|---|---|---|---|---|---|---|
| A | candidate | ring16_acquisition | FAIL | 0/96, 0 | modes 16, mass_tv .021, **hq .716**, **cov_err 3.13**, min_eig 1.30 | 36.4 |
| A | BCAP control | ring16_acquisition | PASS | 26/96, 26 (stable from step 1184) | modes 16, mass_tv .065, hq .961, cov_err .497, min_eig .243 | 30.2 |
| A | NSGDA attribution | ring16_acquisition | FAIL | 0/96, 0 | **modes 15**, mass_tv .081, **hq .458**, **cov_err 54.2**, min_eig 3.87 | 20.0 |
| C | DualNorm + cell objective variant | ring16_acquisition | FAIL | 0/96, 0 | modes 16, mass_tv .034, **hq .742** (best .902 @1450), **cov_err 1.90** (best .960 @1450), min_eig .79 | 43.1 |
| B' | candidate | mode_hold | PASS | 21/24, 21 (stable from 200) | modes 8, hq 1.0 | 26.0 |
| B' | candidate | gaussian1d_smoke | PASS | 18/24, first confirmed 292 | — | 22.3 |
| B' | candidate | gaussian1d_stability | PASS | stationary 72/72, shift-hold 24/24, reacquisition PASS | — | 96.0 |
| B' | BCAP control | mode_hold | FAIL | 10/24, 3 (needs 5) | modes 8, hq .989 at end | 39.0 |
| B' | BCAP control | gaussian1d_smoke | PASS | 3/24, first confirmed 375 | — | 17.4 |
| B' | BCAP control | gaussian1d_stability | FAIL | stationary 2/72, shift-hold 0/24, reacquisition FAIL | — | 58.1 |
| plan | candidate | two_pole | BLOCKED | not runnable | "Cell objective requires the public GANTrainer path" | 0 |
| plan | candidate | five_word_joint_smoke | BLOCKED | not runnable | same blocker | 0 |

`min_eig` is `component_min_eigen_ratio`. Values above 1 mean generated clusters are wider than the target's, not
collapsed. Total training: **388.5 GPU-seconds** (about 6.5 GPU-minutes; the budget was about 2 GPU-hours).

## Failure mode on ring16 (candidate)

Modes and mass are fine. All 16 clusters are found by step ~400 and mass_tv stays at 0.014-0.03. The failure is the
**shape of each cluster**. Generated clusters stay too wide and spill into their neighbours. component_covariance_error
drops to ~3.1 by step 800 and then **plateaus** (3.1-3.2 through step 1600; gate 0.85). hq plateaus at ~0.71 (best
0.736; gate 0.85). component_min_eigen_ratio stays at ~1.3, so clusters are over-dispersed in every direction. The
streak is 0 throughout. This is a converged plateau, not a slow approach, so a longer horizon is unlikely to fix it.

Attribution:

1. The global-NSGDA base alone (the attribution arm) is much worse: covariance error 54, hq 0.46, one mode missed.
   So most of the candidate's ring16 deficit comes from its optimizer swap away from DualNorm, and the cell objective
   recovers a large part of it (cov error 54 -> 3.1, modes 15 -> 16).
2. The cell objective is not neutral on ring16 either. Added to the passing DualNorm control (stage C variant), it
   turns PASS into FAIL. Clusters stay over-wide (min eigen ratio ~0.8 vs control 0.24; cov error >= 0.96 at every
   check vs control 0.46-0.50 at the end). Its best checks (hq .90, cov .96 at step 1450) come close but never pass.

## Conclusion

The "cheapest re-entry probe" named by the zero-pass audit does not re-enter. Matched on one source:

- the cell-quota recipe trades ring16 for mode_hold + gaussian1d_stability, the exact conflict the review describes;
- it is structurally blocked on 2 of the 6 Tier-1 tasks (two_pole and five_word_joint_smoke use particle-cloud priors
  with sigma=0 through the public_components path), so it cannot qualify for Tier 1 in any form;
- moving the cell objective onto the ring16-passing DualNorm base does not rescue ring16 either.

The audit's status for this recipe on ring16, "untried, not shown to conflict", can be updated to "shown to conflict
(matched, seed 0)".

## Caveats

- Seed 0 only, by rule. One run per arm.
- Stage C (DualNorm + cell objective) was a preregistered follow-up (addendum A). Per its stop rule it was not run on
  mode_hold or gaussian1d_stability. So it is unknown whether the cell objective alone, on DualNorm, keeps the held passes.
- Portability: the cell objective uses statistics of observed training data per latent component and needs a
  positive-width, unstandardized learned MoG prior on the public GANTrainer path. It stays within the scope rule
  (trainer/particles only) but is not portable to particle-cloud or component hosts.
- Harness issue on develop (not caused by this change, not fixed here): after every completed job, `forge drain`'s
  post-completion `compile_memory` crashes. First error: `NameError: name 'deepcopy' is not defined` in the cached
  wrappers of `experiments/forge/knowledge.py`. With that import added, the next error is
  `TypeError: _attempts() takes 1 positional argument but 2 were given`. Job results, graded results and receipts persist.
  I re-ran drain per request (a `drainloop` wrapper), so no training was repeated, but boards, readout records and
  compile do not update. That is why the results are in this readout and not in `forge readout` records.
- One control drain was refused once with "another coordinator owns .../coordinator.lock" (two drains on one queue
  root). I re-ran it right after; nothing was lost.
- Attempt receipts live under `/ml2/hypergan/dig-20261015/cell-quota-probe/queue/` and are not committed.

## Evidence

- Preregistration: `/ml2/hypergan/dig-20261015/cell-quota-probe/PREREGISTRATION.md`
- Attempts (graded-result.json, run.log, terminal.json):
  `/ml2/hypergan/dig-20261015/cell-quota-probe/queue/dig-cell-quota-probe-20261015/`
  - ring16: candidate `10b8bd98`, control `f232358d`, attribution `7b425a61`, DualNorm variant `38e4cfa0`
  - held: candidate `583f1c7a` (mode_hold), `d407f6e3` (g1d smoke), `da9e1c84` (g1d stability);
    control `4df01e16`, `4f5aac01`, `7f4a5ee0`
- Drain logs: `/ml2/hypergan/dig-20261015/cell-quota-probe/drain-*.log`
