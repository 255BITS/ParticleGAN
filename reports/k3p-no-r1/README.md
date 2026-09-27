# K3P without R1: what replaces R1's two jobs?

R1 on reals penalizes the correct slope: it pulls grad D(real) toward 0 even when D is right, which flattens D inside
modes. The suspicion is that this is why k3p_simple fails the native toy100 within-mode covariance (accuracy) gate.
R1 does two jobs, spike control and local settling, so this study removes R1 and replaces each job separately.

Base for every arm: `gs2_c03_lr2_d05` from `../k3p-constant-suite/GRID.md`. That is k3p_simple with reg_coeff .3,
lr .0085 and d_lr_mult .5: RpGAN, one-sided RMS fake cap, AMSGrad (0, .999), constant LR, no instance noise, no EMA
anchor, no direct-particle response, spike guard and A2 on. Every arm removes R1 on reals. The fake cap always stays.
Every arm is AMSGrad, so in every arm the spike guard reads `max_exp_avg_sq`, the second moment the update actually
uses. The gs2 reference guard read `exp_avg_sq`, so the control `nr_none_none` differs from the `gs2_c03_lr2_d05`
reference in two ways: R1 is removed and the guard reads a different buffer. Since `max_exp_avg_sq >= exp_avg_sq`,
the guard clips less. In the gs2 runs it clipped 138/267 tensors on ring8-shift/multishift and 3 on
native-staggered100. The guard buffer is the same across all 20 arms, so the settle factor changes only the settling
mechanism.
All terms are in RMS units (d = critic input size) and scaled by `reg_coeff/2`, K3P's convention:

    pen = c/2 * [spike + mean relu(||grad D(f)||/sqrt d - kappa)^2 (+ anchor_weight * prox)]

| spike | term |
|---|---|
| none | nothing (the R1-removed control) |
| symcap | mean relu(\|\|grad D(r)\|\|/sqrt d - kappa)^2 |
| pathcap | the same cap at x = r + u (f - r), u ~ U(0,1) per batch-index pair |
| pairsec | secant slope s_i = \|D(r_i) - D(r_j)\| / \|\|r_i - r_j\|\|, r_j = nearest distinct other real (distance > 1e-6); mean relu(s/sqrt d - kappa)^2 over rows that have one, 0 if none (no double backprop) |
| dvalcap | mean relu(\|D(r) - mean_batch D(r)\| - 1)^2 |

| settle | change |
|---|---|
| none | nothing |
| hinge | critic loss mean relu(1 - (D(r) - D(f))), same pairing; the generator keeps RpGAN softplus |
| oadam | critic update = `lib/oadam.py` OptimisticAdam(amsgrad=True), betas (0, .999), base critic LR |
| anchor | K3P's EMA-anchor prox term (stock weight 1, decay .999) forced on at constant LR, where it is normally gated off (s == 1) |

That gives 20 arms, `nr_<spike>_<settle>` (`gen_arms.py` -> `arms.json`).

## How it runs

`nr_adapter.py` extends `../k3p-constant-suite/suite_adapter.py` without editing it or `particlegan/`. It patches
classes once per task process, so every host sees the same change: native toy100, the standard-trainer and
custom-loop transfer toys, mode_hold hold/shift and the rings. `run_nr.py` reuses the suite's worker and task
protocols unchanged.

    cd reports/k3p-no-r1
    /home/martyn/dev/ParticleGAN/.venv/bin/python run_nr.py --arms all --tasks screen --device cuda:1 --jobs 6
    tail -f logs/*.log              # one line per task
    tail -f logs/<arm>/<task>.log   # one line per eval
    /home/martyn/dev/ParticleGAN/.venv/bin/python summarize_nr.py [--write]   # leaderboard incl. read-only refs

The screen has 7 tasks: native grid100/rotated100/staggered100, toy-img_bars4, toy-two_pole, ring8-shift and
ring8-multishift. The reference results for gs2_c03_lr2_d05, k3p_simple and k3p_stock are read from the
k3p-constant-fix worktree and are not rerun.

## Receipts

Every task directory gets an `nr_receipt.json`, also merged into `result.json` as `nr_receipt`. Every field is
measured during the run, not copied from the config:
- `r1_evals`: every squared input-gradient norm (R1's kernel) on both penalty classes. No nr term requests one.
  `r1_weight_applied` is 0 only when this count is 0.
- `stock_k3p_penalty_calls`: the stock K3P penalty is replaced by a sentinel that counts, then raises.
- the active spike term, with per-term mean and the fraction of calls where it was nonzero. `pairsec_no_neighbor` /
  `pairsec_rows` count the reals dropped for having no distinct neighbour. toy-unused_token_hold feeds 8 identical
  reals, so pairsec is inert there, which makes `nr_pairsec_*` equal to spike=none on that task.
- the loss form and the number of d_loss calls of each form
- per critic optimizer: `oadam_steps` and `adam_steps` (counted per optimizer by the Adam.step dispatcher, which is
  installed in every arm), `lr_receipt_steps` (an independent count from the suite's post-step hook), and the
  `update_rule` derived from those counts. Also `guard_reads`, a tally of the second-moment key the guard actually
  read, plus class, amsgrad, betas, LR and state keys. `oadam_steps_non_critic` must be 0.
- whether the anchor started, at which step, and whether prox was nonzero
- the declared and applied noise stds and LR ranges

`summarize_nr.py` gates each cell on its receipt (`receipt_ok`). The receipt must show the arm's spike/settle, nr
penalty calls > 0, zero R1 evals and zero stock calls, d_loss calls only of the arm's form, OAdam steps only in oadam
arms and only on the critic, the right update rule on every critic, guard reads only of `max_exp_avg_sq`, an active
anchor only in anchor arms, no pathcap pairing mismatch, constant LR and finite terms. A failing cell is shown as
`X <reason>` and not counted as a pass. The leaderboard also lists the tasks where pairsec dropped rows.

Unit tests: `python -m pytest -q reports/k3p-no-r1/test_nr_terms.py tests/test_oadam_amsgrad.py`.

## Smoke (300 steps, `runs_smoke/`)

nr_pathcap_oadam, nr_symcap_anchor and nr_pairsec_hinge ran on native-grid100, toy-two_pole (custom-loop, CPU),
toy-img_bars4 (standard trainer), ring8-shift and toy-unused_token_hold (custom-loop, duplicate reals). All 15
receipts pass `receipt_ok`, and on every route they show:
- 0 R1 evaluations and 0 stock penalty calls.
- In oadam arms, the critic's `oadam_steps` equals its `lr_receipt_steps`, with 0 stock Adam steps and 0 OAdam steps
  on any other optimizer. In the other arms the critic took only stock Adam steps. Every guard read `max_exp_avg_sq`.
- On toy-unused_token_hold, pairsec dropped all 1600/1600 rows (term 0) instead of dividing by a zero distance.
- The anchor started at step 0 (step 1 on the custom-loop hosts, after the critic is identified), and prox was
  nonzero on more than 97% of calls.
- The hinge replaced every critic d_loss call.
- Noise was 0 and the LR was constant.

The transfer toys ignore `--steps`, so those smoke cells are full-budget results.

Caveat: on toy-img_bars4 every cap term (symcap, pathcap, pairsec and the fake cap) was 0 on every call, because
image-critic RMS slopes stay below 1. Without R1, any arm whose only active terms are caps (every spike except
dvalcap, with settle none, hinge or oadam) trains img_bars4 with no gradient penalty at all.

## Results

- `SCREEN.md`: the 20-arm screen on 7 tasks. `AUDIT.md` audits it.
- `SUITE.md`: the full 26-task suite for the screen top 3, plus a reg_coeff 1.0 check (`<arm>_c1`, launched by `full.sh`).
- `EMA_R1.md`: R1 centred on the EMA critic's slope, w * mean ||grad D(r) - beta grad Dbar(r)||^2 / d (settle `emar1`,
  `arms_e1.json`), swept over beta, decay and weight. No beta > 0 beats R1.
