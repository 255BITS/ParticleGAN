# Audit of the no-R1 screen (20 arms x 7 tasks)

**Verdict: PASS.** All 140 cells are valid runs of their arm spec, no cell was overwritten, the SCREEN.md leaderboard
matches the raw results exactly, and the claimed top 3 recompute from raw data. SCREEN.md has four prose
inaccuracies, listed below. None of them changes the verdict or the ranking.

The audit script was independent of `summarize_nr.py` and read only `runs/*/*/{result,nr_receipt,config}.json`,
`bench/{gate,accuracy-gate}-*.json`, `bench/*/progress.log`, `logs/` and the read-only reference runs. Nothing was
rerun and no run file was touched.

## 1. Receipts against the arm spec: 0 issues in 140 cells

Every cell was checked for the following:

| check | result |
|---|---|
| arms.json: noise 0/0, lr .0085, d_lr_mult .5, reg_coeff .3, lr_floor 1, amsgrad on, direct response off, reg_anchor_weight 1 only in anchor arms | all 20 arms |
| receipt arm/spike/settle = arm name; `nr_receipt.json` identical to `result.json["nr_receipt"]` | 140/140 |
| `r1_weight_applied` 0, `r1_evals` 0, `stock_k3p_penalty_calls` 0, nr penalty calls > 0 | 140/140 |
| `penalty_form` = `c/2*[<spike> + fake_cap (+ anchor_weight*prox)]`; spike and fake_cap stats for every call, all finite; spike term exactly 0 for spike=none | 140/140 |
| pairsec rows > 0 with 0 dropped (no duplicate-real task in the screen); pathcap pair mismatch 0 | 140/140 |
| d_loss calls only `rp_hinge` in hinge arms and only `rp_softplus` otherwise; g_loss RpGAN softplus | 140/140 |
| exactly one critic optimizer; update rule `lib.oadam.OptimisticAdam.step` in oadam arms (oadam_steps = lr_receipt_steps, adam_steps 0, `prev_step` in state) and `torch.optim.Adam.step` otherwise; OAdam steps on non-critic optimizers 0 | 140/140 |
| critic amsgrad on (`max_exp_avg_sq` in state), betas (0, .999), LR .00425 (= .0085 x .5) | 140/140 |
| spike guard on, reading only `max_exp_avg_sq` | 140/140 |
| anchor: kernel anchor_weight 1, anchor started and prox > 0 on at least 97.5% of calls in anchor arms; weight 0, not started and no prox elsewhere; kernel coeff .3, kappa 1, s = 1 | 140/140 |
| declared and applied input/output noise 0 (img_bars4's host_policy reports `None`, meaning no noise policy) | 140/140 |
| declared lr .0085 / d_lr_mult .5 / floors 1 / reg_coeff .3; every applied LR group constant, values only in {.0085, .00425, .017}; GANTrainer recipes match | 140/140 |
| `config.json` lr / d_lr_mult / noise / floors / reg_coeff agree with the arm | 140/140 |
| `steps_override` None (full budget, no smoke leakage); only the 7 screen task dirs per arm | 140/140 |

The code matches the spec as well. `lib/oadam.py` differs from d78a5a05 only by the opt-in amsgrad flag, which keeps
torch's AMSGrad convention (the max of the raw v, then bias correction) and leaves the lookback unchanged.
`particlegan/` is unchanged. The stock anchor weight is 1.0 (`recipes.py`) and the decay is .999.

## 2. Continuity: no task dir was overwritten

- Every `logs/<arm>.log` has one start header, one `finished` footer and exactly 7 task lines, each task once, all
  with rc=0. `screen_done.txt` shows both launchers exited 0 after a 08:18:49 to 09:20:45 run, about 62 minutes.
- Every `logs/<arm>/<task>.log` has one `# arm=` header and one `# DONE` line, and its eval steps strictly increase.
  Its DONE status matches `result.json`. Every native `bench/*/progress.log` has one START and strictly increasing
  TRAIN steps.
- Every file in every task dir has an mtime inside its own arm's start/finish window, and no file is newer than that
  cell's `result.json`. Each `result.json` mtime is within 3 s of the cell's line in the arm log.
- The committed result files are byte-identical to the working tree (`git diff HEAD` is empty). The bench artifacts
  (npz, snapshots, events) are untracked. The 20 `.launcher.lock` files are empty per-arm lock files, which is
  expected.
- The reference `result.json` files were written between 2026-09-26 22:48 and 2026-09-27 01:49, before the screen
  started, so the other agent's activity did not change them during the screen.

## 3. SCREEN.md numbers against the raw data

The 23-row leaderboard table regenerates byte-identical from `summarize_nr.py` and equals LEADERBOARD.md. The
independent recompute agrees on passes, ring fails (38-656), native passes (0 of 60 nr native cells), rotated100 HQ
without an anchor (.851-.940), img_bars4 (only `nr_dvalcap_anchor` passes, 9/24), the settle-factor means
(anchor 123.6/15.8, oadam 355.4/9.0, none 399.2/7.2, hinge 404.0/4.0), the spike-factor means (none 244.5,
pathcap 242.3, symcap 310.8, pairsec 369.3, dvalcap 436.0; toy cells 8.75 for every term except dvalcap, which has 10.0),
the worst covariance of the control and references (nr_none_none .150, gs2 .142, k3p_simple .144), and the zero cap
terms on img_bars4 (the symcap/pathcap/pairsec spike term and the fake cap are 0.0 on every img_bars4 call).

These are the prose errors, from most to least significant:

1. **"both hinge arms without a value term fail two_pole" is wrong.** Three hinge arms fail two_pole: symcap_hinge
   (3/24), pathcap_hinge (4/24) and pairsec_hinge (3/24). `nr_none_hinge`, which has no value term, passes 5/24.
   Read correctly, the claim is that hinge combined with a gradient or secant cap fails two_pole.
2. **"eigenvalue range stays about 0.1-2.3 ... the same shape that gs2 has" understates the damage.** The final
   per-mode eigen-ratio range is 0.000-2.91 for the non-anchor arms and 0.000-3.10 over all 20 arms. Two cells
   collapse to an eigen ratio of 0: pairsec_oadam rotated100 (.0005) and dvalcap_oadam staggered100 (0.0). The
   median worst-case minimum among non-anchor arms is .068, against gs2's .142. Only the control is on par with gs2.
   This strengthens the conclusion that removing R1 does not help covariance, because it usually makes the covariance
   worse.
3. **"The anchor ... worst cov ratio of 0.01-0.02 on every native task" is too strong.** The per-arm worst value is
   .0099-.0159, but the per-task minima reach .041-.049 on grid100 (none/pathcap anchor) and staggered100
   (symcap anchor).
4. **The spike-term figures are slightly off.** "Active on 86-100% of calls on the ring" holds only for the
   non-anchor arms (99.6-100%). With the anchor on, the positive fraction is .56-.86 (pairsec_anchor multishift .557,
   symcap_anchor multishift .725, pathcap_anchor multishift .819). The symcap native term mean spans .0004-.024,
   not .0007-.024: symcap_anchor staggered100 is .0004.

## 4. Recomputed top selection

The SCREEN rule is native passes, then toy suffix sum (img_bars4 + two_pole), then worst native min eigen-ratio,
higher being better:

| rank | arm | native passes | toy sum | worst min eig | ring fails | screen passes | cov err |
|---|---|---|---|---|---|---|---|
| 1 | nr_dvalcap_anchor | 0 | 23 | .0159 | 228 | 2/7 | 4.00 |
| 2 | nr_none_anchor | 0 | 14 | .0125 | 38 | 1/7 | 3.97 |
| 3 | nr_pathcap_anchor | 0 | 14 | .0113 | 52 | 1/7 | 3.75 |
| 4 | nr_pairsec_anchor | 0 | 14 | .0107 | 112 | 1/7 | 4.44 |
| 5 | nr_symcap_anchor | 0 | 14 | .0099 | 188 | 1/7 | 4.09 |

This matches the claimed top `["nr_dvalcap_anchor", "nr_none_anchor", "nr_pathcap_anchor"]`. Ranks 2-5 tie on the
first two keys, and the third key separates pathcap (#3) from pairsec (#4) by only .0006. The leaderboard's own rule
(passes, then ring fails) gives the same three arms in the same order, and pathcap beats pairsec on ring fails
(52 vs 112), so the selection is robust. A ring-only ranking would swap `nr_dvalcap_anchor` (228) for
`nr_pairsec_anchor` (112). SCREEN.md already says that no top arm is an R1 replacement, and all of them are worse
than gs2 except on two_pole.
